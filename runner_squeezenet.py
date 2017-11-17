import tensorflow as tf
from scipy.misc import imsave

import model.squeezenet as squeezenet
import squeezenet_helper
from image_utils import load_image
from app import App

if __name__ == '__main__':
    SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--content_image", required=True)
    # parser.add_argument("--content_image_size", default=None)
    # parser.add_argument("--style_image", required=True)
    # parser.add_argument("--style_image_size", default=None)
    # parser.add_argument("--output_image", required=True)
    #
    # args = parser.parse_args()
    # content_image_path = args.content_image
    # content_image_size = args.content_image_size
    # style_image_path = args.style_image
    # style_image_size = args.style_image_size
    # output_path = args.output_image

    config = {
        'init_random': True,
        'decaying_lr': True,
        'learning_rate': 1e-1,
        'decay_at': [500],
        'lr_values': [1e-2],

        'num_iterations': 500,
        'content_layers': [3],
        'style_layers': [1, 3, 4, 6],  # , 7, 9],
        'style_weights': [1, 1, 1, 1, 1, 1],  # [21 * 100, 50, 5, 5, 7, 7],
        'content_weight': 0,  # 1e-3,
        'style_weight': 1,
        # 'tv_weight': 0,  # 1e-2
        'summary_directory': 'summary/squeezenet'
    }

    # content_image = image_utils.load_image(content_image_path, size=content_image_size)
    # style_image = image_utils.load_image(style_image_path, size=style_image_size)

    content_image = load_image("styles/blank.jpg", size=512)
    style_image = load_image("styles/the_scream.jpg")

    session = tf.Session()

    model = squeezenet.SqueezeNet(SAVE_PATH, session)
    model_helper = squeezenet_helper.SqueezeNetHelper()
    app = App(model, model_helper, config, session)

    generated_image = app.style_transfer(content_image, style_image)

    # imsave(output_path, generated_image)
    imsave("generated/style_1346_w1_s512.png", generated_image)
