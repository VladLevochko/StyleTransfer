import image_utils
import tensorflow as tf
from base_style_transfer import StyleTransferBase
from losses.content_loss import ContentLoss
from losses.mrf_style_loss import MrfBasedStyleLoss
from losses.total_variation_loss import TotalVariationLoss
from nets import vgg


class SemanticMrfBasedStyleTransfer(StyleTransferBase):
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

        self.content_sem_path = "styles/Seth_sem.png"
        self.style_sem_path = "styles/Gogh_sem.png"

    def build_graph(self):
        print("[.] building graph")

        content_features = self.get_content_features(self.content_image)
        style_features = self.get_style_features_with_semantic(self.style_image, self.style_sem_path)

        self.x = tf.Variable(tf.random_uniform(self.content_image.shape, 0, 1, dtype=tf.float32), name="x")
        x_content_features = self.get_content_features(self.x)
        x_style_features = self.get_style_features_with_semantic(self.x, self.content_sem_path)

        self.content_loss_v = self.content_loss.get_value(content_features, x_content_features)
        self.style_loss_v = self.style_loss.get_value(style_features, x_style_features)
        self.tv_loss_v = self.total_variation_loss.get_value(self.x)

        self.loss_v = self.content_weight * self.content_loss_v + \
                      self.style_weight * self.style_loss_v + \
                      self.tv_weight * self.tv_loss_v

        with tf.variable_scope("optimizer") as opt_scope:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_v, var_list=[self.x])
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)

        self.session.run(tf.variables_initializer([self.learning_rate, self.x] + opt_vars))

        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="features")
        rv_dict = {}
        for var in restore_variables:
            name = var.name
            name = name.replace("features/", "")
            name = name.split(":")[0]
            rv_dict[name] = var

        saver = tf.train.Saver(var_list=rv_dict)
        saver.restore(self.session, self.checkpoint_path)

        print("[.] graph built")

    def get_style_features_with_semantic(self, image, sem_path):
        style_features = self.get_features(image, self.style_layers)
        semantic_features = image_utils.load_image_pil(sem_path)  #TODO: clip values of semantic features to [0, 1]
        for i in range(len(style_features)):
            sub_features = style_features[i]
            shape = sub_features.shape
            local_semantic_features = self.downscale_semantic_features(semantic_features, shape[1], shape[2])
            style_features[i] = tf.concat([sub_features, local_semantic_features], 3)

        return style_features

    def downscale_semantic_features(self, semantic_features, height, width):
        rescaled = tf.image.resize_bicubic(semantic_features[None], (height, width))
        return rescaled


if __name__ == "__main__":
    content_image_path = "styles/Seth.jpg"
    style_image_path = "styles/Gogh.jpg"

    content_image = image_utils.load_image_pil(content_image_path, 256)
    style_image = image_utils.load_image_pil(style_image_path, 256)

    content_image = image_utils.preprocess(content_image)
    style_image = image_utils.preprocess(style_image)

    st = SemanticMrfBasedStyleTransfer()
    generated = st.run(content_image, style_image)
    generated = image_utils.deprocess(generated)

    print()
