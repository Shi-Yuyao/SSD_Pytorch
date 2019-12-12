import torch
import torch.nn as nn
import torch.nn.functional as F


def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    # conv_layer conv8_2
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
    # conv_layer conv9_2
    layers += [nn.Conv2d(512, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
    # conv_layer conv10_2
    # layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    # layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    # conv_layer conv11_2
    # layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    # layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False, nlin_layer=nn.ReLU):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False, nlin_layer=Hswish):
    nr = nn.ReLU6 if onnx_compatible else nlin_layer
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nr(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nr(inplace=True)
        )


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, size=300, n_class=1000, input_size=224, width_mult=1.0, dropout=0.8, mode='large',
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV3, self).__init__()
        bottleneck = MobileBottleneck
        input_channel = 16
        last_channel = 1280
        self.size = size
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c, se,    nl,  s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 672, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c, se,   nl,  s
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(bottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # do SSD in nn.Sequential
        self.extras = nn.ModuleList(
            add_extras(str(self.size), last_channel))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.mean(3).mean(2)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x):
        sources = list()
        for i in range(14):
            x = self.features[i](x)

        # Main Block 1 in SSD, matched size = 480
        for i in range(3):  # 执行主干网络的第15层，执行bottelneck的第一部分（共三部分）升维
            x = self.features[14].conv[i](x)
        sources.append(x)  # 将该层的feature map作为第一层，加入SSD中

        for i in range(3, len(self.features[14].conv)):
            x = self.features[14].conv[i](x)

        x = self.features[15](x)
        # Main Block 2 in SSD, matched size = 960
        x = self.features[16](x)
        sources.append(x)  # 将该层的feature map作为第二层，加入SSD中

        for i in range(17, len(self.features)):
            x = self.features[i](x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)  # 将extra的奇数层的feature map加入SSD中，extra共4层，SSD一共收到6层

        return sources

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def SSDMobilenetv3(size, channel_size='48'):
    return MobileNetV3(size=size)


if __name__ == '__main__':
    import os

    '''临时创建主函数检测网络结构'''
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    modelv3 = SSDMobilenetv3(size=512)
    checkpoint = torch.load('../weights/pretrained_models/ssd_mobilenet_v3_large.pth')
    modelv3.features.load_state_dict(checkpoint)
    with torch.no_grad():
        modelv3.eval()
        x = torch.randn(1, 3, 512, 512)
        modelv3(x)
        import time

        st = time.time()
        for i in range(1):
            modelv3(x)
        print(time.time() - st)
