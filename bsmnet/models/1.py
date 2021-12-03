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

        # self.compress1_2 = convbn(64, 32, 3, 1, 1, 1)
        self.compress1_4 = convbn(128, 32, 3, 1, 1, 1)
        # self.lastconv = nn.Sequential(convbn(224, 128, 3, 1, 1, 1),
        #                                   nn.ReLU(inplace=True),
        #                                   nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1,
        #                                                      output_padding=1, bias=True))

        self.lastconv = nn.Sequential(convbn(224, 128, 3, 1, 1, 1),
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
        # print(t_a.shape,t_b.shape,out1_1.shape)
        out1_1 = torch.cat((t_a, t_b, out1_1), 1)
        out1_1 = self.lastconv(out1_1)

        # print("out2_1",out2_1.shape,"out1_1",out1_1.shape,"out1_2",out1_2.shape,"out1_4",out1_4.shape)
        return {"out1_1": out1_1, "out1_4": self.compress1_4(out1_4)}


class Multiscale_fusion(nn.Module):
    def __init__(self, inplanes=16, ker=3):
        super(Multiscale_fusion, self).__init__()

        self.inplane = inplanes
        self.ker = ker

        self.firstconv = nn.Sequential(convbn_3d(inplanes * 2, inplanes, 3, 1, 1),
                                       nn.ReLU(inplace=True))

        self.conv3D_fusion = nn.Sequential(convbn_3d(inplanes, inplanes, (3, 1, 1), (3, 1, 1), (0, 0, 0)),
                                           nn.ReLU(inplace=True),
                                           convbn_3d(inplanes, inplanes * 2, 3, 1, 1),
                                           nn.ReLU(inplace=True))

        self.scale = nn.Sequential(convbn(1 + self.inplane * 2, 16, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(16, 16, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(16, 1, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention = nn.Sequential(convbn(1, 16, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(16, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, ker * inplanes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.unfold = nn.Unfold((3, 1), 1, (1, 0), 1)

    def forward(self, x, depth):
        input = self.firstconv(x)
        b, c, d, h, w = input.shape
        uncertainty_feature = torch.cat((depth, x[:, :, 2, :, :]), 1)
        scale = self.scale(uncertainty_feature)
        attention = self.attention(scale)
        # div = 3*self.inplane
        # wei = attention[:,:div,:,:]
        # bias = attention[:,div:,:,:]

        input = input.view(b, c, d, -1)
        # print(self.unfold(input).shape,input.shape)
        input = self.unfold(input).view(b, c * 3, d, h, w)

        intermedium = (torch.unsqueeze(attention, 2) * input).view(b, c, 3 * d, h, w)
        out = self.conv3D_fusion(intermedium)
        # print(intermedium.shape,out.shape)

        return out, scale


class GwcNet(nn.Module):
    def __init__(self, maxdepth):
        super(GwcNet, self).__init__()
        self.maxdepth = maxdepth

        self.feature_extraction = feature_extraction()

        self.dres00 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True))
        self.dres01 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(32, 32, 3, 1, 1))

        self.refine1 = Multiscale_fusion()
        self.refine2 = Multiscale_fusion()

        # self.dre11 = refinementHourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 16, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 16, kernel_size=3, padding=1, stride=1, bias=False))

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

    def forward(self, left, right, gt, baseline, focus):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        left_volume1_4 = features_left["out1_4"].unsqueeze(2).repeat(1, 1, self.maxdepth // 4, 1, 1)
        dis_plane = generate_dis(1, self.maxdepth, 4, 4, baseline, focus)
        right_volume1_4 = build_right(features_right["out1_4"], dis_plane, self.maxdepth // 4)
        volume = torch.cat((left_volume1_4, right_volume1_4), 1)

        cost0 = self.dres00(volume)
        cost0 = self.dres01(cost0) + cost0
        cost0 = self.classif0(cost0)

        cost0 = F.upsample(cost0, [self.maxdepth, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        pred0 = F.softmax(cost0, dim=1)
        pred0 = depth_regression(pred0, 1, self.maxdepth, 1)
        # print(pred0)

        coar_dis = torch.unsqueeze(baseline * focus / (pred0 + 0.00000001), 1)

        right_volume1_1 = volume_warping(features_right["out1_1"], coar_dis, 5)

        left_volume = features_left["out1_1"].unsqueeze(2).repeat(1, 1, 5, 1, 1)
        volume1_1 = torch.cat((left_volume, right_volume1_1), 1)

        sca = generate_scale(gt)

        volume1, scale1 = self.refine1(volume1_1, torch.unsqueeze(pred0, 1))
        cost1 = self.classif1(volume1)
        P1 = torch.exp(-torch.pow(cost1, 2).sum(1) / (2 * torch.pow(scale1, 2) + 0.000000001))
        P1 = P1 / (scale1 * 2.51 + 0.00000001)
        # cost1 = torch.squeeze(cost1, 1)
        cost1 = F.softmax(P1, dim=1)
        pred1 = depth_regress(cost1, pred0, scale1)

        volume2, scale2 = self.refine2(volume1, torch.unsqueeze(pred1, 1))
        cost2 = self.classif1(volume2)
        P2 = torch.exp(-torch.pow(cost2, 2).sum(1) / (2 * torch.pow(scale2, 2) + 0.000000001))
        P2 = P2 / (scale2 * 2.51 + 0.00000001)
        # cost2 = torch.squeeze(cost2, 1)
        cost2 = F.softmax(P2, dim=1)
        pred2 = depth_regress(cost2, pred1, scale2)

        del volume
        del right_volume1_1
        del right_volume1_4
        del left_volume
        del left_volume1_4

        if self.training:
            # print(sca.shape, scale1.shape)
            return [pred0, pred1, pred2]
            # ,[torch.squeeze((scale1-sca).abs(),1), torch.squeeze(((scale2-sca/2.0)).abs(),1)]

        else:

            return [pred0, pred1, pred2]


def GwcNet_G(d):
    return GwcNet(d)


def GwcNet_GC(d):
    return GwcNet(d)
