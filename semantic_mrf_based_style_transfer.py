import image_utils
import tensorflow as tf
import tensorflow.contrib.slim as slim
from base_style_transfer import StyleTransferBase
from losses.content_loss import ContentLoss
from losses.mrf_style_loss import MrfBasedStyleLoss
from losses.total_variation_loss import TotalVariationLoss
from nets import vgg
from fcn.fcn_16s import FCN_16s


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

        self.content_weight = 1
        self.style_weight = 5
        self.semantic_weight = 1
        self.tv_weight = 0

        self.learning_rate = tf.Variable(5e-1, name="learning_rate")
        self.num_iterations = 50

        self.content_layers = ["conv4_2"]
        self.style_layers = ["conv3_1", "conv4_1"]
        self.model = vgg.vgg_19
        self.model_arg_scope = vgg.vgg_arg_scope()

        self.train_op = None
        self.summaries = None
        self.writer = None

        self.content_image = None
        self.style_image = None

        self.checkpoint_path = "checkpoints/vgg_19.ckpt"
        self.semantic_model_checkpoint_path = "/home/vladyslav/projects/tf-image-segmentation/fcn_16s_checkpoint/model_fcn16s_final.ckpt"
        self.number_of_classes = 21

        self.content_semantic_features = None
        self.style_semantic_features = None

    def build_graph(self):
        print("[.] building graph")

        self.precompute_semantic_features()

        content_features = self.get_content_features(self.content_image)
        style_features = self.get_style_features(self.style_image)
        style_features = self.concat_semantic_features(style_features, self.style_semantic_features)

        self.x = tf.Variable(tf.random_uniform(self.content_image.shape, 0, 1, dtype=tf.float32), name="x")
        x_content_features = self.get_content_features(self.x)
        x_style_features = self.get_style_features(self.x)
        x_style_features = self.concat_semantic_features(x_style_features, self.content_semantic_features)

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

    def concat_semantic_features(self, style_features, semantic_features):
        map = {}
        prev = tf.expand_dims(semantic_features, 0)
        for i in range(5):
            if i == 0:
                pooled = prev
            else:
                pooled = slim.max_pool2d(prev, [2, 2])
            map[i + 1] = pooled
            prev = pooled

        i = 0
        for style_layer in self.style_layers:
            key = int(style_layer[4])
            style_features[i] = tf.concat([style_features[i], map[key]], axis=3)
            i += 1

        return style_features

    def precompute_semantic_features(self):
        semantic_features, _ = FCN_16s(tf.stack([self.content_image, self.style_image]),
                                       self.number_of_classes, is_training=False)

        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="fcn_16")
        saver = tf.train.Saver(var_list=restore_variables)
        saver.restore(self.session, self.semantic_model_checkpoint_path)
        computed_semantic_features = self.session.run(semantic_features)

        self.content_semantic_features = computed_semantic_features[0]
        self.style_semantic_features = computed_semantic_features[1]


if __name__ == "__main__":
    content_image_path = "styles/Seth.jpg"
    style_image_path = "styles/Gogh.jpg"

    size = (320, 256)
    content_image, original_content_image_size = image_utils.load_image_pil(content_image_path, size)
    style_image, _ = image_utils.load_image_pil(style_image_path, size)

    content_image = image_utils.preprocess(content_image)
    style_image = image_utils.preprocess(style_image)

    st = SemanticMrfBasedStyleTransfer()
    generated = st.run(content_image, style_image)
    generated = image_utils.deprocess(generated)

    image_utils.save_generated_image(generated, "summary", "generated_image_name_stub", original_content_image_size)
