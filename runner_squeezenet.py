import cs231n.image_utils as image_utils
import cs231n.classifiers.squeezenet as squeezenet
import tensorflow as tf
import squeezenet_helper
import matplotlib.pyplot as plt
from app import App


if __name__ == '__main__':
    SAVE_PATH = 'cs231n/datasets/squeezenet.ckpt'

    config = {
        'init_random': True,
        'learning_rate': 9e-1,
        'decayed_lr': 4e-1,
        'decay_at': 100,
        'num_iterations': 500,
        # 'content_layers': [3],
        'content_layers': [1, 2, 3],
        # 'style_layers': [1, 3,  4, 6],
        'style_layers': [1, 2, 3, 4, 6, 7, 9],
        'content_weights': [7cs , 7, 7],
        # 'style_weights': [256, 128, 2, 1],
        'style_weights': [21, 21, 1, 1, 1, 7, 7],
        'tv_weight': 0,  # 1e-2
        'summary_directory': 'summary/squeezenet'
    }
    content_image = image_utils.load_image("styles/tubingen.jpg", size=256)
    style_image = image_utils.load_image("styles/starry_night.jpg", size=512)

    session = tf.Session()

    model = squeezenet.SqueezeNet(SAVE_PATH, session)
    model_helper = squeezenet_helper.SqueezeNetHelper()
    app = App(model, model_helper, config, session)

    generated_image = app.style_transfer(content_image, style_image)

    f, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[2].set_title('Stylized Image')
    axarr[0].imshow(content_image)
    axarr[1].imshow(style_image)
    axarr[2].imshow(generated_image)
    plt.show()
