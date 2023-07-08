from torch.utils.data import DataLoader


def load_dataset(data, batchsize: int, is_shuffle: bool = True):
    """
    Dataloader
    """
    return DataLoader(dataset=data, batch_size=batchsize, shuffle=is_shuffle)
