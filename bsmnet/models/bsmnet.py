from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.compress1_4 = convbn(128, 32, 3, 1, 1, 1)

        self.lastconv = nn.Sequential(convbn(227, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                      convbn(128, 16, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.firstconv(x)
        out1_1 = fea

        fea = self.layer1(fea)
        l2 = self.layer2(fea)
        out1_2 = l2

        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        out1_4 = l4

        t_a = F.upsample(out1_4, [x.size()[2], x.size()[3]], mode='bilinear')
        t_b = F.upsample(out1_2, [x.size()[2], x.size()[3]], mode='bilinear')
       
        out1_1 = torch.cat((t_a,t_b,out1_1,x),1)
        out1_1 = self.lastconv(out1_1)

        return {"out1_1": out1_1, "out1_4":self.compress1_4(out1_4)}


class Multiscale_fusion(nn.Module):
    def __init__(self,inplanes = 16,ker = 3):
        super(Multiscale_fusion, self).__init__()

        self.inplane = inplanes
        self.ker = ker

        self.firstconv = nn.Sequential(convbn_3d(inplanes*2, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3D_fusion = nn.Sequential(convbn_3d(inplanes, inplanes, (3,1,1), (3,1,1), (0,0,0)),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes*2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.scale = nn.Sequential(convbn(inplanes*2+1, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 16, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(16, 1, 3, 1, padding=1, bias=False),
                                        nn.ReLU(inplace=True))


        self.attention = nn.Sequential(convbn(inplanes*2+1, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 16, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(16, ker*inplanes, 3, 1, padding=1))

        self.unfold = nn.Unfold((3,1), 1, (1,0), 1)


 
    def forward(self, x, depth):

        uncertainty_feature =  x[:, :, 2, :, :]
        input = self.firstconv(x)

        scale = self.scale(torch.cat((depth,uncertainty_feature),1))
        # scale_temp = torch.ones_like(scale)*5

        attention = self.attention(torch.cat((scale,uncertainty_feature),1))
        attention = F.sigmoid(attention)

        # attention_temp = torch.ones_like(attention)
        # attention_temp = F.sigmoid(attention_temp)

        b,c,d,h,w = input.shape
        input = input.view(b,c,d,-1)
        
        input = self.unfold(input).view(b,c*3,d,h,w)

        intermedium = (torch.unsqueeze(attention, 2)*input).view(b,c,3*d,h,w)
        out = self.conv3D_fusion(intermedium)

        return out, scale

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, (1,2,2), 1),
                                   nn.ReLU(inplace=True))


        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, (1,2,2), 1),
                                   nn.ReLU(inplace=True))


        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels))

    def forward(self, x):
        conv1 = self.conv1(x)

        conv3 = self.conv3(conv1)

        conv5 = F.relu(self.conv5(conv3) + conv1, inplace=True)
        conv6 = F.relu(self.conv6(conv5) + x, inplace=True)

        return conv6



class BSMNet(nn.Module):
    def __init__(self, maxdepth):
        super(BSMNet, self).__init__()
        self.maxdepth = maxdepth


        self.feature_extraction = feature_extraction()

        self.dres00 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres01 = hourglass(32)


        self.refine1 = Multiscale_fusion()
        # self.refine2 = Multiscale_fusion()

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, gt, baseline,focus):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        depth_interval=1
        left_volume1_4 = features_left["out1_4"].unsqueeze(2).repeat(1, 1, self.maxdepth//depth_interval, 1, 1)
        dis_plane = generate_dis(1,self.maxdepth,depth_interval,4,baseline,focus)
        right_volume1_4 = build_right(features_right["out1_4"],dis_plane, self.maxdepth//depth_interval)
        volume = torch.cat((left_volume1_4,right_volume1_4),1)

        cost0 = self.dres00(volume)
        cost0 = self.dres01(cost0)
        cost0 = self.classif0(cost0)

        cost0 = torch.squeeze(cost0, 1)
        cost0 = F.upsample(cost0, [left.size()[2], left.size()[3]], mode='bilinear')
        
        pred0 = F.softmax(cost0, dim=1)
        pred0 = depth_regression(pred0, 1, self.maxdepth, depth_interval)

        coar_dis = torch.unsqueeze(baseline*focus/(pred0.detach()+0.00000001),1)

        right_volume1_1 = volume_warping(features_right["out1_1"], coar_dis, 5)

        left_volume = features_left["out1_1"].unsqueeze(2).repeat(1, 1, 5, 1, 1)
        volume1_1 = torch.cat((left_volume,right_volume1_1),1)
       
        volume1,scale1 = self.refine1(volume1_1,torch.unsqueeze(pred0,1))
        cost1 = self.classif1(volume1)
        cost1 = torch.squeeze(cost1, 1)
        cost1 = F.softmax(cost1, dim=1)
        pred1 = depth_regress(cost1, pred0, scale1)


        # del volume
        # del right_volume1_1
        # del right_volume1_4
        # del left_volume
        # del left_volume1_4


        if self.training:
            return [pred0,pred1],scale1

        else:
            return [pred0, pred1],scale1


def bsm_net(d):
    return BSMNet(d)



