from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from src.data.datasets.utils.image.randaug import RandAugment

from transformers import AutoFeatureExtractor

inception_normalize = dict(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
)  # This is simple maximum entropy normalization performed in Inception paper
imagenet_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
clip_normalize = dict(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

NORMALIZATIONS = {
    "inception": inception_normalize,
    "imagenet": imagenet_normalize,
    "clip": clip_normalize,
}
INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def get_normalize_kwargs_and_interpolation_mode(
    normalization: str = None, interpolation: str = None, pretrained_vision_model: str = None
):
    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(pretrained_vision_model) if pretrained_vision_model else None
    )
    if (normalization is None or interpolation is None) and feature_extractor is None:
        raise ValueError(
            "If pretrained_vision_model is not specified, normalization and interpolation must be both specified"
        )
    return _get_normalize_kwargs(normalization, feature_extractor), _get_interpolation_mode(
        interpolation, feature_extractor
    )


def _get_normalize_kwargs(normalization: str, feature_extractor):
    if normalization is None:
        normalize_kwargs = dict(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    else:
        normalize_kwargs = NORMALIZATIONS[normalization]
    return normalize_kwargs


def _get_interpolation_mode(interpolation: str, feature_extractor):
    if interpolation is None:
        interpolation_mode = _interpolation_modes_from_int(feature_extractor.resample)
    else:
        interpolation_mode = INTERPOLATIONS[interpolation]
    return interpolation_mode


def center_crop_transform(
    size: int,
    normalization: str = None,
    interpolation: str = None,
    pretrained_vision_model: str = None,
    tensorize: bool = True,
    do_normalization: bool = True,
):
    normalize_kwargs, interpolation_mode = get_normalize_kwargs_and_interpolation_mode(
        normalization, interpolation, pretrained_vision_model
    )
    return Compose(
        [
            Resize(size, interpolation=interpolation_mode),
            CenterCrop(size),
            convert_to_rgb,
        ]
        + ([ToTensor()] if tensorize else [])
        + ([Normalize(**normalize_kwargs)] if do_normalization else [])
    )


def center_crop_transform_randaug(
    size: int,
    normalization: str = None,
    interpolation: str = None,
    pretrained_vision_model: str = None,
    tensorize: bool = True,
    do_normalization: bool = True,
):
    # TODO: Is there a reason to use RGBA? Currently just keeping it because it was used in METER
    trs = (
        [convert_to_rgb, RandAugment(2, 9), convert_to_rgba]
        if "clip" in pretrained_vision_model
        else [RandAugment(2, 9)]
    )
    trs.append(
        center_crop_transform(
            size,
            normalization=normalization,
            interpolation=interpolation,
            pretrained_vision_model=pretrained_vision_model,
            tensorize=tensorize,
            do_normalization=do_normalization,
        )
    )
    return Compose(trs)


def resize_transform(
    size: int,
    normalization: str = None,
    interpolation: str = None,
    pretrained_vision_model: str = None,
    tensorize: bool = True,
    do_normalization: bool = True,
):
    normalize_kwargs, interpolation_mode = get_normalize_kwargs_and_interpolation_mode(
        normalization, interpolation, pretrained_vision_model
    )
    return Compose(
        [Resize((size, size), interpolation=interpolation_mode)]
        + ([convert_to_rgb] if "clip" in pretrained_vision_model else [])
        + ([ToTensor()] if tensorize else [])
        + ([Normalize(**normalize_kwargs)] if do_normalization else [])
    )


def convert_to_rgb(image):
    return image.convert("RGB")


def convert_to_rgba(image):
    return image.convert("RGBA")


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
