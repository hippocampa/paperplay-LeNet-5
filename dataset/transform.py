import torchvision


transform_fn = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Pad(2),
        torchvision.transforms.Normalize(mean=0, std=1),
    ]
)
