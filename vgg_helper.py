import numpy as np


class VggHelper:

    def __init__(self):
        self.mean = np.array([123.6800, 116.7790, 103.9390])

    def preprocess_image(self, image):
        """Preprocess an image for squeezenet.

        Subtracts the pixel mean and divides by the standard deviation.
        """
        return image #- self.mean

    def deprocess_image(self, image, rescale=False):
        """Undo preprocessing on an image and convert back to uint8."""
        return image #+ self.mean
