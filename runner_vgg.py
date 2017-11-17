import time
import argparse
import tensorflow as tf
import vgg_helper
import model.vgg19 as vgg19
import matplotlib.pyplot as plt
from image_utils import load_image
from app import App

parser = argparse.ArgumentParser()
parser.add_argument("--content_image", required=True, help="")
parser.add_argument("--style_image", required=True, help="")
parser.add_argument("--num_iterations", default=500, help="")
parser.add_argument("--learning_rate", default=1e-1, help="")
parser.add_argument("--decaying_rate", default=False, help="")
parser.add_argument("--content_laysers", default=[9], help="")
parser.add_argument("--style_layers", default=[])


if __name__ == '__main__':
    SAVE_PATH = 'model/imagenet-vgg-verydeep-19.mat'

    config = {
        'init_random': True,
        'decaying_lr': True,
        'learning_rate': 1e-1,
        'decay_at': [500],
        'lr_values': [1e-2],

        'num_iterations': 1000,
        'content_layers': [9],
        'style_layers': [0, 3, 6, 11, 16],
        'style_weights': [2e-1, 2e-1, 2e-1, 2e-1, 2e-1],
        'content_weight': 0,
        'style_weight': 1e4,
        # 'tv_weight': 0,  # 1e-2
        'summary_directory': 'summary/vgg19'
    }
    content_image = load_image("styles/tubingen.jpg", size=256)
    style_image = load_image("styles/stary_night.jpg", size=256)

    session = tf.Session()

    model = vgg19.VGG19(SAVE_PATH, session)
    model_helper = vgg_helper.VggHelper()
    app = App(model, model_helper, config, session)

    start = time.clock()
    generated_image = app.style_transfer(content_image, style_image)
    end = time.clock()

    print("elapsed time: {}".format(end - start))

    plt.imsave("/home/vlad/PycharmProjects/style_transfer_application/generated/vgg/tub_sn.png", generated_image)
