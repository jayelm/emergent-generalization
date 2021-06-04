from . import feature
from . import vision

BACKBONES = {
    "resnet18": vision.ResNet18,
    "conv4": vision.Conv4,
}
