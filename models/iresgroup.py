import torch
import torch.nn as nn
import os
from div.download_from_url import download_from_url

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pretrained')

__all__ = ['iResGroup', 'iresgroup50', 'iresgroup101', 'iresgroup152']


model_urls = {
    'iresgroup50': 'https://drive.google.com/uc?export=download&id=1mCcUKTLbJfjLr2ZiUZPT89wlNf6WxWr_',
    'iresgroup101': 'https://drive.google.com/uc?export=download&id=1i_btVQzQeXFgHFfDi5nTEb_fb7gCtqLG',
    'iresgroup152': 'https://drive.google.com/uc?export=download&id=1rXDMj-yWcKEiO3LILLZlarh69HEibXMc',
}


def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResGroupBlock(nn.Module):
    reduction = 2

    def __init__(self, inplanes, planes,groups, stride=1, downsample=None, norm_layer=None,
                 start_block=False, end_block=False, exclude_bn0=False):
        super(ResGroupBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if not start_block and not exclude_bn0:
            self.bn0 = norm_layer(inplanes)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups=groups, stride=stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes // self.reduction)

        if start_block:
            self.bn3 = norm_layer(planes // self.reduction)

        if end_block:
            self.bn3 = norm_layer(planes // self.reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class iResGroup(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, groups=None,  dropout_prob0=0.0):
        super(iResGroup, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups is None:
            groups = 64

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 256, layers[0], groups=max(1, groups//8), stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 512, layers[1], groups=max(1, groups//4), stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 1024, layers[2], groups=max(1, groups//2), stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 2048, layers[3], groups=groups, stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        self.fc = nn.Linear(2048 // block.reduction, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResGroupBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, groups, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 and self.inplanes != planes // block.reduction:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes // block.reduction),
                norm_layer(planes // block.reduction),
            )
        elif self.inplanes != planes // block.reduction:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes // block.reduction),
                norm_layer(planes // block.reduction),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample, norm_layer,
                            start_block=True))
        self.inplanes = planes // block.reduction
        exclude_bn0 = True
        for _ in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes, groups, norm_layer=norm_layer,
                                exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(block(self.inplanes, planes, groups, norm_layer=norm_layer, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)

        return x



def iresgroup50(pretrained=False, **kwargs):
    """Constructs a iResGroup-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResGroup(ResGroupBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresgroup50'],
                                                           root=default_cache_path)))
    return model


def iresgroup101(pretrained=False, **kwargs):
    """Constructs a iResGroup-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResGroup(ResGroupBlock, [3, 4, 23, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresgroup101'],
                                                           root=default_cache_path)))
    return model


def iresgroup152(pretrained=False, **kwargs):
    """Constructs a iResGroup-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = iResGroup(ResGroupBlock, [3, 8, 36, 3], **kwargs)
    if pretrained:
        os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load(download_from_url(model_urls['iresgroup152'],
                                                           root=default_cache_path)))
    return model

