import PIL.BmpImagePlugin
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

__all__ = [
    "opencv2pil", "opencv2tensor", "pil2opencv", "process_image"
]


def opencv2pil(image: np.ndarray):
    """ OpenCV Convert to PIL.Image format.
    Returns:
        PIL.Image.
    """

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def opencv2tensor(image: np.ndarray, gpu: int):
    """ OpenCV Convert to torch.Tensor format.
    Returns:
        torch.Tensor.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nhwc_image = torch.from_numpy(rgb_image).div(255.0).unsqueeze(0)
    input_tensor = nhwc_image.permute(0, 3, 1, 2)
    if gpu is not None:
        input_tensor = input_tensor.cuda(gpu, non_blocking=True)
    return input_tensor


def pil2opencv(image: PIL.BmpImagePlugin.BmpImageFile):
    """ PIL.Image Convert to OpenCV format.
    Returns:
        np.ndarray.
    """

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def process_image(image: PIL.BmpImagePlugin.BmpImageFile, gpu: int = None):
    """ PIL.Image Convert to PyTorch format.
    Args:
        image (PIL.BmpImagePlugin.BmpImageFile): File read by PIL.Image.
        gpu (int): Graphics card model.
    Returns:
        torch.Tensor.
    """
    tensor = transforms.ToTensor()(image)
    input_tensor = tensor.unsqueeze(0)
    if gpu is not None:
        input_tensor = input_tensor.cuda(gpu, non_blocking=True)
    return input_tensor
