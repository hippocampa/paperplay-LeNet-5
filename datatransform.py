from torchvision import transforms


def transform_fn() -> transforms.Compose:
    t = transforms.Compose(transforms.Normalize(mean=0, std=1))
    return t
