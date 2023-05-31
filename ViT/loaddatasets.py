import os
import json
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def build_transform(is_train):
    if is_train:
        transform = create_transform(
            input_size=384,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        return transform

    t = []
    size = int(384 / 0.875)
    t.append(
        transforms.Resize(size, interpolation=3), 
    )
    t.append(transforms.CenterCrop(384))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)