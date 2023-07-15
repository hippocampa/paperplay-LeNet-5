import fire
from dataset import data
from dataset import load_data


def main(batchsize: int = 1):
    train_set, test_set = data.get_data()
    train_loader, test_loader = load_data.loader(
        train_set, test_set, batchsize=batchsize
    )
    print(iter(train_loader).next()[0].shape)


if __name__ == "__main__":
    fire.Fire(main)
