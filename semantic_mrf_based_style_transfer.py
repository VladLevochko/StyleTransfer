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

        self.name = "soft_masks"

        self.x = None
        self.content_loss = ContentLoss()
        self.style_loss = MrfBasedStyleLoss()
        self.total_variation_loss = TotalVariationLoss()

        self.content_weight = 1
        self.style_weight = 1
        self.semantic_weight = 1  # it doesn't work yet
        self.tv_weight = 1

        self.learning_rate_value = 6e-1
        self.learning_rate = tf.Variable(self.learning_rate_value, name="learning_rate")
        self.num_iterations = 100

        self.content_layers = ["conv4_2"]
        self.style_layers = ["conv4_1", "conv5_1"]
        self.model = vgg.vgg_19
        self.model_arg_scope = vgg.vgg_arg_scope()
        self.checkpoint_path = "checkpoints/vgg_19.ckpt"

        self.number_of_classes = 21
        self.content_semantic_features = None
        self.style_semantic_features = None
        self.semantic_model_checkpoint_path = "checkpoints/fcn_16s_checkpoint/model_fcn16s_final.ckpt"

    def build_graph(self):
        print("[.] building graph")

        self.precompute_semantic_features()

        content_features = self.get_content_features(self.preprocessed_content_image)
        style_features = self.get_style_features(self.preprocessed_style_image)
        style_features = self.append_semantic_features(style_features, self.style_semantic_features)

        self.x = tf.Variable(tf.random_uniform(self.content_image.shape, 0, 1, dtype=tf.float32), name="x")
        x_content_features = self.get_content_features(self.x)
        x_style_features = self.get_style_features(self.x)
        x_style_features = self.append_semantic_features(x_style_features, self.content_semantic_features)

        with tf.variable_scope("content_loss"):
            self.content_loss_v = self.content_weight * \
                                  self.content_loss.get_value(content_features, x_content_features)
        with tf.variable_scope("style_loss"):
            self.style_loss_v = self.style_weight * \
                                self.style_loss.get_value(style_features, x_style_features)
        with tf.variable_scope("total_variation_loss"):
            self.tv_loss_v = self.tv_weight * \
                             self.total_variation_loss.get_value(self.x)

        with tf.variable_scope("total_loss"):
            self.loss_v = self.content_loss_v + self.style_loss_v + self.tv_loss_v

        with tf.variable_scope("optimizer") as opt_scope:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_v, var_list=[self.x])
        self.opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)

        print("[.] graph built")

    def append_semantic_features(self, style_features, semantic_features):
        means = tf.reduce_mean(semantic_features, axis=[0, 1])
        _, indices = tf.nn.top_k(means, k=5)
        semantic_features = tf.gather(semantic_features, indices, axis=2)
        # semantic_features = tf.transpose(semantic_features, [1, 2, 0])

        with tf.name_scope("semantic_features_downsampling"):
            map = {}
            prev = tf.expand_dims(semantic_features, 0)
            for i in range(5):
                if i == 0:
                    pooled = prev
                else:
                    pooled = slim.avg_pool2d(prev, [2, 2])
                map[i + 1] = pooled
                prev = pooled

        with tf.name_scope("semantic_features_appending"):
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

        with tf.name_scope("semantic_features_precomputation"):
            with tf.variable_scope("content_semantic_features"):
                self.content_semantic_features = computed_semantic_features[0]
            with tf.variable_scope("style_semantic_features"):
                self.style_semantic_features = computed_semantic_features[1]


if __name__ == "__main__":
    content_image_path = "styles/cars/golf7r.jpg"
    style_image_path = "styles/car_drawing.jpg"

    size = (333, 229)
    content_image, original_content_image_size = image_utils.load_image_pil(content_image_path, size)
    style_image, _ = image_utils.load_image_pil(style_image_path, size)

    # content_image = image_utils.preprocess(content_image)
    # style_image = image_utils.preprocess(style_image)

    st = SemanticMrfBasedStyleTransfer()
    generated = st.run(content_image, style_image)
    # generated = image_utils.deprocess(generated)

    image_utils.save_generated_image(generated, "summary", "generated_image_name_stub", original_content_image_size)
