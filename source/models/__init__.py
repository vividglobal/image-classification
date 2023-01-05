from .mobilenetv2 import MobileNetV2

from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152,\
                    resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,\
                    wide_resnet101_2
from .shufflenetv2 import shufflenet_v2_x0_5,shufflenet_v2_x1_0
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}
model_fn = {
    'mobilenet_v2': MobileNetV2,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
}
