import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
# from scipy.misc import imread, imresize

from PIL import Image

"""
Utility functions used for viewing and processing images.
"""

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# def image_from_url(url):
#     """
#     Read an image from a URL. Returns a numpy array with the pixel data.
#     We write the image to a temporary file then read it back. Kinda gross.
#     """
#     try:
#         f = urllib.request.urlopen(url)
#         _, fname = tempfile.mkstemp()
#         with open(fname, 'wb') as ff:
#             ff.write(f.read())
#         img = imread(fname)
#         os.remove(fname)
#         return img
#     except urllib.error.URLError as e:
#         print('URL Error: ', e.reason, url)
#     except urllib.error.HTTPError as e:
#         print('HTTP Error: ', e.code, url)

#
# def load_image(filename, size=None):
#     """Load and resize an image from disk.
#
#     Inputs:
#     - filename: path to file
#     - size: size of shortest dimension after rescaling
#     """
#     img = imread(filename)
#     if size is not None:
#         orig_shape = np.array(img.shape[:2])
#         min_idx = np.argmin(orig_shape)
#         scale_factor = float(size) / orig_shape[min_idx]
#         new_shape = (orig_shape * scale_factor).astype(int)
#         img = imresize(img, scale_factor)
#     return img


def load_image_pil(filename, size=None):
    """Load and resize an image from disk using PIL library.

    Args:
        filename (str): Path to file.
        size (int): Size of shortest dimension after rescaling.

    Returns:
        np.array: Pixel grid of image.
    """
    image = Image.open(filename)

    if size is not None:
        original_shape = image.size
        scale_factor = size / np.min(original_shape)
        new_shape = (int(original_shape[0] * scale_factor), int(original_shape[1] * scale_factor))
        image = image.resize(new_shape)

    return np.array(image)


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


def save_generated_image(image, summary_directory, name="image"):
    # files = os.listdir(summary_directory)
    # target_directory = sorted(files)[-1]
    image_name = os.path.join(summary_directory, name + ".png")

    res = Image.fromarray(image)
    res.save(image_name)


if __name__ == "__main__":
    path = "styles/la_muse.jpg"
    image = load_image_pil(path, 256)
    image = preprocess(image)
    image
