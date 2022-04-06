import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def ConvBNReLU(in_chann, out_chann, ks=3, st=1, p=1, with_act=True):
    return nn.Sequential(
        nn.Conv2d(in_chann, out_chann, ks, st, p, bias=False), 
        nn.BatchNorm2d(out_chann), 
        nn.ReLU(inplace=True) if with_act else Identity()
    )

class DWConv(nn.Module):
    def __init__(self, in_chans, out_chans, ks=3, st=1, p=1):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=ks, stride=st, 
                      padding=p, groups=in_chans, bias=False), 
            nn.BatchNorm2d(in_chans), 
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chans), 
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x 

class PostionAtten(nn.Module):
    def __init__(self, in_chans):
        super(PostionAtten, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ave_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([ave_feat, max_feat], dim=1)
        y = self.sigmoid(self.conv(y))

        return x*y + x

class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.s1 = nn.Sequential(
            DWConv(3, 64, 3, 2, 1), 
            DWConv(64, 64, 3, 1, 1)
        )

        self.s2 = nn.Sequential(
            DWConv(64, 96, 3, 2, 1),
            DWConv(96, 96, 3, 1, 1), 
            DWConv(96, 96, 3, 1, 1)
        )

        self.s3 = nn.Sequential(
            DWConv(96, 128, 3, 2, 1), 
            DWConv(128, 128, 3, 1, 1), 
            DWConv(128, 128, 3, 1, 1)
        )

        # 空间注意力
        self.pa = PostionAtten(128)

        self._init_weights()

    def forward(self, x):
        feat1 = self.s1(x)
        feat2 = self.s2(feat1)
        feat3 = self.s3(feat2) 
        feat3 = self.pa(feat3)
        return feat3

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

class SEModule(nn.Module):
    def __init__(self, in_chann):
        super(SEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Linear(in_chann, in_chann // 6, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_chann//6, in_chann, bias=True), 
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.pool(x).view(bs, c)
        y = self.conv(y).view(bs, c, 1, 1)
        return x * y

class ChannelAtten(nn.Module):
    def __init__(self, in_chann):
        super(ChannelAtten, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Linear(in_chann, in_chann // 6, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_chann//6, in_chann, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.pool(x).view(bs, c)
        y = self.conv(y).view(bs, c, 1, 1)
        return x * y + x

class GELayers1(nn.Module):
    def __init__(self, in_chann, out_chann, padding=1, dilation=1):
        super(GELayers1, self).__init__()
        mid_chann = in_chann * 6
        self.conv1 = ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chann, mid_chann, kernel_size=3, stride=1, 
                      padding=padding, groups=in_chann, dilation=dilation, bias=False), 
            nn.BatchNorm2d(mid_chann), 
            SEModule(mid_chann), 
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chann, out_chann, kernel_size=1, 
                      stride=1,  bias=False), 
            nn.BatchNorm2d(out_chann)
        )
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat) 
        feat = self.conv2(feat)
        out = feat + x
        
        return self.relu(out)

class GELayers2(nn.Module):
    def __init__(self, in_chann, out_chann, padding=1, dilation=1):
        super(GELayers2, self).__init__()
        mid_chann = in_chann * 6
        self.conv1 = ConvBNReLU(in_chann, in_chann, 3, 1, 1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_chann, mid_chann, kernel_size=3, stride=2, 
                      padding=1, groups=in_chann, bias=False), 
            nn.BatchNorm2d(mid_chann), 
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chann, mid_chann, kernel_size=3, stride=1, 
                      padding=padding, groups=mid_chann, dilation=dilation, bias=False), 
            nn.BatchNorm2d(mid_chann), 
            SEModule(mid_chann), 
            nn.ReLU(inplace=True), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chann, out_chann, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chann)
        )

        self.short = nn.Sequential(
            nn.Conv2d(in_chann, in_chann, 3, 2, 1, groups=in_chann, 
                      bias=False), 
            nn.BatchNorm2d(in_chann), 
            nn.Conv2d(in_chann, out_chann, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(out_chann)
        )

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)

        out = feat + self.short(x)
        return self.relu(out)

class Stem(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Stem, self).__init__()
        mid_chans = out_chans // 2
        self.conv = ConvBNReLU(in_chans, out_chans, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBNReLU(out_chans, mid_chans, 1, 1, 0), 
            ConvBNReLU(mid_chans, out_chans, 3, 2, 1), 
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.last_conv = ConvBNReLU(out_chans*2, out_chans, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        left_feat = self.left(feat)
        right_feat = self.right(feat)
        out = torch.cat([left_feat, right_feat], dim=1)
        out = self.last_conv(out)
        
        return out

class CEBlockSimAspp(nn.Module):
    def __init__(self, in_channs, rates=(3, 6, 9)):
        super(CEBlockSimAspp, self).__init__()
        mid_chans = in_channs//2
        self.conv = ConvBNReLU(in_channs, mid_chans, 1, 1, 0) # n
        self.convs = nn.ModuleList()
        self.convs.append(ConvBNReLU(mid_chans, mid_chans, 1, 1, 0, with_act=False))
        for rate in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(mid_chans, mid_chans, 3, 1, padding=rate, 
                              dilation=rate, bias=False),
                    nn.BatchNorm2d(mid_chans), 
                )
            )
        self.convs.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                ConvBNReLU(mid_chans, mid_chans, 1, 1, 0, with_act=False) 
            )
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_chans*5, in_channs, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(in_channs), 
            nn.ReLU(inplace=True) 
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        feat = self.conv(x)
        outs = []
        for conv in self.convs:
            outs.append(conv(feat))
        d0 = outs[0]
        d1 = d0 + outs[1]
        d2 = d1 + outs[2]
        d3 = d2 + outs[3]
        d4 = d3 + outs[4]
        feat = torch.cat([d0, d1, d2, d3, d4], dim=1)
        out = self.project(feat)
        return out

class ARM(nn.Module):
    def __init__(self, in_chans):
        super(ARM, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(in_chans), 
            nn.Sigmoid()
        )
    def forward(self, x):
        identity = x
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = self.process(x)
        x = torch.mul(x, identity)
        return x + identity

class FuseModule(nn.Module):
    def __init__(self, in_chans32, in_chans16):
        super(FuseModule, self).__init__()
        self.arm16 = ARM(64)
        self.conv16 = ConvBNReLU(in_chans16, in_chans32, 3, 1, 1)

        self.arm32 = ARM(128)
        self.conv32 = ConvBNReLU(in_chans32, in_chans16, 3, 1, 1)
    def forward(self, feat32, feat16):
        avg = torch.mean(feat32, dim=[2, 3], keepdim=True)
        feat32 = self.arm32(feat32)
        feat32 = feat32 + avg
        feat32 = F.interpolate(feat32, size=feat16.size()[2:], 
                               mode="bilinear", align_corners=True)
        feat32 = self.conv32(feat32) 

        feat16 = self.arm16(feat16)
        feat16 = feat32 + feat16
        feat16 = F.interpolate(feat16, size=(2*feat16.size(2), 2*feat16.size(3)), 
                               mode="bilinear", align_corners=True)
        feat16 = self.conv16(feat16) 
        return feat16, feat32 # 1/8, 1/16

class SegmentBranch(nn.Module):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.s1s2 = Stem(3, 16)
        self.s3 = nn.Sequential(
            GELayers2(16, 32), 
            GELayers1(32, 32), 
        )
        
        self.s4 = nn.Sequential(
            GELayers2(32, 64), 
            GELayers1(64, 64), 
        )

        self.s5_4 = nn.Sequential(
            GELayers2(64, 128), 
            GELayers1(128, 128), 
            GELayers1(128, 128), 
            GELayers1(128, 128), 
        )
        self.s5_5 = CEBlockSimAspp(128)
        self.cma = ChannelAtten(128)
        self.fuse = FuseModule(128, 64)

    def forward(self, x):
        feat2 = self.s1s2(x) # 1/4
        feat3 = self.s3(feat2) # 1/8
        feat4 = self.s4(feat3) # 1/16
        feat5_4 = self.s5_4(feat4) # 1/32
        feat5_5 = self.s5_5(feat5_4) # 1/32
        # 通道自注意力
        feat5_5 = self.cma(feat5_5)

        feat8, feat16 = self.fuse(feat5_5, feat4)
        return feat8, feat16

class FFM(nn.Module):
    def __init__(self, in_channels):
        super(FFM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        identity = x
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = torch.mul(identity, x) + identity
        return x

class SimBGABlock(nn.Module):
    def __init__(self, detail_chans, seg_chans):
        super(SimBGABlock, self).__init__()
        self.detail_branch = DWConv(detail_chans, seg_chans, 3, 1, 1)
        self.seg_branch_left = DWConv(seg_chans, seg_chans, 3, 1, 1)
        self.seg_branch_right = DWConv(seg_chans, seg_chans, 3, 1, 1)

        self.final_conv = DWConv(seg_chans*2, detail_chans, 3, 1, 1)
        self.ffm = FFM(detail_chans)
        self._init_weights()

    def _init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, detail, context):
        detail_feat = self.detail_branch(detail)

        segout_feat_left = self.seg_branch_left(context)
        segout_feat_right = self.seg_branch_right(context)

        detail = segout_feat_left.sigmoid() * detail_feat
        final_feat = torch.cat([detail, segout_feat_right], dim=1)
        final_feat = self.ffm(self.final_conv(final_feat))
        return final_feat

class SegmentHead(nn.Module):
    def __init__(self, in_chann, mid_chann, num_classes, up_factor=8):
        super(SegmentHead, self).__init__()
        self.up_factor = up_factor
        self.conv = ConvBNReLU(in_chann, mid_chann, 3, 1, 1)
        self.drop = nn.Dropout(0.1)

        self.classifier = nn.Conv2d(mid_chann, num_classes, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(self.up_factor*x.size(2), self.up_factor*x.size(3)), 
                            mode="bilinear", align_corners=True)
        return x
