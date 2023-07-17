import fire
import torch
from PIL import Image
from dataset import transform
from model import LeNet5


PATH = "savedmodels/mark1.pt"


def get_label(fpath: str) -> str:
    """
    Get image label of a dir:
    dir/label/img.jpg

    Args:
        fpath: Path of an image.

    Return:
        Image label
    """
    fname = fpath.split("/")[-2]
    return fname


def makepred(file: str) -> None:
    label = get_label(file)
    image = Image.open(file)
    transformed_img = transform.transform_fn(image)
    LeNet = LeNet5()
    LeNet.load_state_dict(torch.load(PATH))
    LeNet.eval()

    with torch.inference_mode():
        X = torch.unsqueeze(transformed_img, dim=0).to("cpu")
        y_pred = LeNet(X)
        print(f"Actual number: {label}")
        print(f"Predicted result: {y_pred.argmax().item()}")


if __name__ == "__main__":
    fire.Fire(makepred)
