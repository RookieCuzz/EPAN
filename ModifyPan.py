import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torchvision.ops import DeformConv2d

from . import arch_util



class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels,kernel_size, stride=1, padding=1):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )

    def forward(self, x):
        return self.depthwise_conv(x)

# class MyModel(nn.Module):
#     def __init__(self, input_channels):
#         super(MyModel, self).__init__()
#         self.layer1 = DepthwiseConv2d(input_channels, kernel_size=3, stride=1, padding=1)
#         self.layer2 = nn.Conv2d(input_channels, 64, kernel_size=1)  # Pointwise convolution
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.relu(x)
#         return x
#
# # Example usage
# input_channels = 3
# model = MyModel(input_channels)
#
# input_tensor = torch.randn(1, input_channels, 32, 32)
# output_tensor = model(input_tensor)
#
# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output_tensor.shape}")
class ESCPA(nn.Module):
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(ESCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Conv2d(group_width, group_width,3, stride=1,padding=1)
        self.FDB=FDB(group_width,group_width)
        # self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.FDB(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out
class FDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FDB, self).__init__()
        self.fdu1=FeatureDistillationUnit(in_channels)
        self.fdu2=FeatureDistillationUnit(in_channels)
        self.fdu3=FeatureDistillationUnit(in_channels)
        self.cca=VarianceChannelAttention(in_channels)
        self.fc1=nn.Conv2d (int(in_channels*1.75), in_channels, 1)
        self.fc2=nn.Conv2d (in_channels,in_channels,3,1,1)
    def forward(self, x):
        d1,r1=self.fdu1(x)
        d2,r2=self.fdu2(r1)
        d3,r3=self.fdu3(r2)
        output=torch.cat((d1, d2, d3,r3), dim=1)
        output=self.cca(self.fc1(output))
        output=self.fc2(output+output)
        return output
class VarianceChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(VarianceChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        var_out = self.fc(self.variance_pool(x).view(x.size(0), -1))
        out = var_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x

    def variance_pool(self, x):
        # 计算方差
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=(2, 3), keepdim=True)
        return var

class FeatureDistillationUnit(nn.Module):
    def __init__(self, in_channels,distillation_rate=0.25):
        super(FeatureDistillationUnit, self).__init__()
        # 计算蒸馏特征和剩余特征的通道数
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        #蒸馏加工
        self.conv0 = nn.Conv2d(self.distilled_channels, self.distilled_channels, kernel_size=1)
        #通道膨胀
        self.conv1 = nn.Conv2d(self.remaining_channels, in_channels, kernel_size=1)
        self.epa = EPA(in_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 定义卷积层和批量归一化层
        # self.conv = nn.Conv2d(in_channels, self.remaining_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(self.remaining_channels)

    def forward(self, x):
        # 将输入特征分为蒸馏特征和剩余特征
        distilled_features = x[:, :self.distilled_channels, :, :]
        remaining_features = x[:, self.distilled_channels:, :, :]
        # 对剩余特征进行卷积和激活
        remaining_features=self.lrelu(self.epa(self.conv1(remaining_features)))
        # remaining_features = F.relu(self.bn(self.conv(remaining_features)))
        #对蒸馏的特征进行加工
        distilled_features=self.conv0(distilled_features)
        return distilled_features, remaining_features


class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class EPA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(EPA, self).__init__()
        self.convPath1 = nn.Conv2d(nf//2, nf//2, 1)
        self.convPath2= nn.Conv2d(nf //2, nf//2, 1)
        self.sigmoid = nn.Sigmoid()
        self.depthwise_conv1 = DepthwiseConv2d(in_channels=nf//2, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv2= DepthwiseConv2d(in_channels=nf//2, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv3= DepthwiseConv2d(in_channels=nf//2,kernel_size=3, stride=1, padding=1)

        # self.depthwise_conv4= DepthwiseConv2d(nf, nf,kernel_size=3,padding=1)
        self.depthwise_conv5=DepthwiseConv2d(in_channels=nf, kernel_size=3, stride=1, padding=1)
        self.conv1x1= nn.Conv2d(nf, nf, kernel_size=1)
    def forward(self, x):
        part1_default, part2_default = torch.chunk(x, 2, dim=1)
        part1_step1 = self.depthwise_conv1(part1_default)
        part1_step2=self.depthwise_conv2(part1_step1+part1_default)
        part1_step3=self.convPath1(part1_step2+part1_step1)

        part2_step1 = self.depthwise_conv3(part2_default)
        part2_step2=self.convPath2(part2_step1+part2_default)

        merged_output = torch.cat((part1_step3, part2_step2), dim=1)
        merged_output=self.sigmoid(merged_output)
        out = torch.mul(self.conv1x1(x + self.depthwise_conv5(x)), merged_output)

        return out





class EPAN(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=4):
        super(EPAN, self).__init__()
        # ESCPA
        ESCPA_block_f = functools.partial(ESCPA, nf=nf, reduction=2)
        self.scale = scale

        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        ### main blocks
        self.SCPA_trunk = arch_util.make_layer(ESCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = EPA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = EPA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out
# # # Example usage
# input_channels = 64
# model = EPAN(input_channels)
#
#
#
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")