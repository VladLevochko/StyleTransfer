import numpy as np


class SqueezeNetHelper:

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image):
        """Preprocess an image for squeezenet.

        Subtracts the pixel mean and divides by the standard deviation.
        """
        return (image.astype(np.float32) / 255.0 - self.mean) / self.std

    def deprocess_image(self, image, rescale=False):
        """Undo preprocessing on an image and convert back to uint8."""
        image = (image * self.std + self.mean)
        if rescale:
            vmin, vmax = image.min(), image.max()
            image = (image - vmin) / (vmax - vmin)
        return np.clip(255 * image, 0.0, 255.0).astype(np.uint8)
