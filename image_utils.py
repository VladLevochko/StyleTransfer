import os
import numpy as np
from PIL import Image

"""
Utility functions used for viewing and processing images.
"""

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_pil(filename, size=None):
    """Load and resize an image from disk using PIL library.

    Args:
        filename (str): Path to file.
        size (tuple): Size of shortest dimension after rescaling.

    Returns:
        np.array: Pixel grid of image.
    """
    image = Image.open(filename)
    original_size = image.size

    if size is not None:
        desired_shape = size
        actual_shape = (desired_shape[0] - desired_shape[0] % 32,
                        desired_shape[1] - desired_shape[1] % 32)
        image = image.resize(actual_shape)

    return np.array(image), original_size


def preprocess(image):
    """Subtract mean and divide by standard deviation of ImageNet.

    Args:
        image (np.array): Pixel grid of image.

    Returns:
        np.array: Processed pixel grid.
    """
    return (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def deprocess(image):
    """

    Args:
        image:

    Returns:

    """
    image = (image * IMAGENET_STD + IMAGENET_MEAN)
    return np.clip(255 * image, 0.0, 255.0).astype(np.uint8)


def save_generated_image(image, summary_directory, name="image", size=None):
    image_name = os.path.join(summary_directory, name + ".png")

    res = Image.fromarray(image)
    if size is not None:
        res = res.resize(size)
    res.save(image_name)
