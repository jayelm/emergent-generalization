import torch
from torchvision import transforms
from PIL import Image, ImageEnhance


transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_t_(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def unnormalize_t_(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Unnormalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(
        self,
        aug=False,
        normalize=True,
        to_pil=True,
    ):
        if aug:
            transform_list = [
                "RandomResizedCrop",
                "ImageJitter",
                "RandomHorizontalFlip",
                "ToTensor",
            ]
        else:
            transform_list = ["Resize", "CenterCrop", "ToTensor"]

        if normalize:
            transform_list.append("Normalize")

        if to_pil:
            transform_list = ["ToPILImage"] + transform_list

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    def get_normalize(self):
        return self.parse_transform("Normalize")
