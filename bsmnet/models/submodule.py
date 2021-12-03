from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def depth_regression(x, min,max,interval,depth_values=None):
    assert len(x.shape) == 4
    if depth_values is None:
        depth_values = torch.arange(min, max + 1, interval,dtype=x.dtype, device=x.device).view(1, max//interval, 1, 1)

    # print(x.shape,(x * depth_values).shape,depth_values.shape)
    return torch.sum(x * depth_values, 1, keepdim=False)

def depth_regress(x, pred,scale):
    assert len(x.shape) == 4
    b,d,h,w=x.shape

    depth_values = torch.arange(-2, 3, dtype=x.dtype, device=x.device)
    depth_values = depth_values.view(1, 5, 1, 1)
    # print(x.shape)
    offset = torch.sum(x * depth_values, 1, keepdim=False)*torch.squeeze(scale,1)


    return offset+pred

def generate_dis(min,max,interval,scale,baseline,focus):
    depth_range = torch.arange(min,max+1,interval).view(max//interval,1,1)
    # depth_range = depth_range.view(max,1,1).repeat(1,h,w)
    depth_plane = (baseline*focus/depth_range)/scale
    return depth_plane.cuda().float()

def generate_depth(min,max,interval,scale,coadis):
    depth_range = torch.arange(min,max+1,interval,dtype=coadis.dtype, device=coadis.device).view(1,max*2+1,1,1)*scale+coadis
    depth_range = baseline*focus/(depth_range+0.00000001)
    # depth_range = depth_range.view(max,1,1).repeat(1,h,w)
    # depth_plane = (1050*0.27/depth_range)/scale
    # print(depth_range.shape,depth_range)
    return depth_range.cuda().float()

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



def volume_warping(src_feature, coa_depth, num_depth):
    # Apply homography warpping on one src feature map from src to ref view.
    batch, channels, height, width = src_feature.shape

    with torch.no_grad():

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y))  # [2, H*W]
        xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # [B, 2, H*W]
        offset = torch.arange(-2,3,dtype=xyz.dtype, device=xyz.device).view(1,5,1,1)

        match_depth = coa_depth+offset
        match_depth = match_depth.view(batch,5,height * width)
        # match_depth = torch.cat((match_depth, torch.zeros_like(match_depth)),1) #b,2,5,h*w
        volume_xyz = xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) #b,2,num,h*w

        x_normalized = ((volume_xyz[:, 0, :, :]-match_depth) / ((width - 1) / 2) - 1).clamp(-1,1)
        y_normalized = volume_xyz[:, 1, :, :] / ((height - 1) / 2) - 1
        xy_normalized = torch.stack((x_normalized, y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = xy_normalized

    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def build_right(src_feature, depth_plane, num_depth):
    # build original candidate right feature volume.

    batch, channels,height, width = src_feature.shape
     # = src_feature.shape[2], src_feature.shape[3]

    with torch.no_grad():

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_feature.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_feature.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y))  # [2, H*W]
        xyz = xyz.unsqueeze(0).repeat(batch, 1, 1)  # [B, 2, H*W]

        volume_xyz = xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) #b,2,num,h*w
        # volume_xyz[:, 0, :, :] = volume_xyz+depth # b,2,num,h*w
        x_normalized = ((volume_xyz[:, 0, :, :]-depth_plane.view(num_depth,1)) / ((width - 1) / 2) - 1).clamp(-1,1)
        y_normalized = volume_xyz[:, 1, :, :] / ((height - 1) / 2) - 1
        xy_normalized = torch.stack((x_normalized, y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = xy_normalized.cuda()
  
    warped_src_fea = F.grid_sample(src_feature, grid.view(batch, num_depth *height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def generate_scale(gt):
    scale = torch.zeros_like(gt)
    # print(gt.shape)
    mask = (gt <= 10.0)
    scale[mask] = 5/100.0

    mask = (gt > 10)&(gt <= 20)
    scale[mask] = 10 / 100.0

    mask = (gt > 20) & (gt <= 30)
    scale[mask] = 20 / 100.0

    mask = (gt > 30) & (gt <= 40)
    scale[mask] = 40 / 100.0

    mask = (gt > 40) & (gt <= 50)
    scale[mask] = 80 / 100.0

    mask = (gt > 50) & (gt <= 60)
    scale[mask] = 160 / 100.0

    mask = (gt > 60) & (gt <= 70)
    scale[mask] = 320 / 100.0

    mask = (gt > 70)
    scale[mask] = 640 / 100.0




    return torch.unsqueeze(scale,1)
