import image_utils
import tensorflow as tf
from base_style_transfer import StyleTransferBase
from losses.content_loss import ContentLoss
from losses.mrf_style_loss import MrfBasedStyleLoss
from losses.total_variation_loss import TotalVariationLoss
from nets import vgg


class MrfBasedStyleTransfer(StyleTransferBase):
    def __init__(self):
        super().__init__()

        self.session = tf.Session()

        self.x = None
        self.content_loss = ContentLoss()
        self.style_loss = MrfBasedStyleLoss()
        self.total_variation_loss = TotalVariationLoss()

        self.content_loss_v = 0
        self.style_loss_v = 0
        self.tv_loss_v = 0
        self.loss_v = 0

        self.content_weight = 10
        self.style_weight = 1
        self.tv_weight = 0.1

        self.learning_rate = tf.Variable(1e-1, name="learning_rate")
        self.num_iterations = 30

        self.content_layers = ["conv4_2"]
        self.style_layers = ["conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        self.model = vgg.vgg_19
        self.model_arg_scope = vgg.vgg_arg_scope()

        self.train_op = None
        self.summaries = None
        self.writer = None

        self.content_image = None
        self.style_image = None

        self.checkpoint_path = "checkpoints/vgg_19.ckpt"


if __name__ == "__main__":
    content_image_path = "styles/tubingen.jpg"
    style_image_path = "styles/la_muse.jpg"

    content_image = image_utils.load_image_pil(content_image_path, 256)
    style_image = image_utils.load_image_pil(style_image_path, 256)

    content_image = image_utils.preprocess(content_image)
    style_image = image_utils.preprocess(style_image)

    st = MrfBasedStyleTransfer()
    generated = st.run(content_image, style_image)
    generated = image_utils.deprocess(generated)

    print()
