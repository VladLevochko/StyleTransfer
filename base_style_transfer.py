import os
import json
import time
import tensorflow as tf
import image_utils
import tensorflow.contrib.slim as slim


class StyleTransferBase:
    def __init__(self):
        self.session = None

        self.name = "base_style_transfer"

        self.x = None
        self.content_loss = None
        self.style_loss = None
        self.total_variation_loss = None

        self.content_loss_v = 0
        self.style_loss_v = 0
        self.tv_loss_v = 0
        self.loss_v = 0

        self.content_weight = 0
        self.style_weight = 0
        self.tv_weight = 0

        self.learning_rate_value = 0
        self.learning_rate = tf.Variable(self.learning_rate_value, name="learning_rate")
        self.num_iterations = 0

        self.content_layers = []
        self.style_layers = []
        self.model = None
        self.model_arg_scope = None

        self.train_op = None
        self.summaries = None
        self.writer = None

        self.content_image = None
        self.style_image = None

        self.opt_vars = None
        self.checkpoint_path = "checkpoint_path_stub"

        self.summary_path = "summary_path_stub"

    def run(self, content_image, style_image):
        self.content_image = content_image
        self.style_image = style_image

        self.build_graph()
        self.initialize_variables()
        self.prepare_summaries()
        self.run_style_transfer()

        x = self.session.run(self.x)

        return x

    def build_graph(self):
        print("[.] building graph")

        content_features = self.get_content_features(self.content_image)
        style_features = self.get_style_features(self.style_image)

        self.x = tf.Variable(tf.random_uniform(self.content_image.shape, 0, 1, dtype=tf.float32), name="x")
        x_content_features = self.get_content_features(self.x)
        x_style_features = self.get_style_features(self.x)

        self.content_loss_v = self.content_loss.get_value(content_features, x_content_features)
        self.style_loss_v = self.style_loss.get_value(style_features, x_style_features)
        self.tv_loss_v = self.total_variation_loss.get_value(self.x)

        self.loss_v = self.content_weight * self.content_loss_v + \
                      self.style_weight * self.style_loss_v + \
                      self.tv_weight * self.tv_loss_v

        with tf.variable_scope("optimizer") as opt_scope:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_v, var_list=[self.x])
        self.opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)

        print("[.] graph built")

    def initialize_variables(self):
        print("[.] initializing variables")

        self.session.run(tf.variables_initializer([self.learning_rate, self.x] + self.opt_vars))

        restore_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="features")
        rv_dict = {}
        for var in restore_variables:
            name = var.name
            name = name.replace("features/", "")
            name = name.split(":")[0]
            rv_dict[name] = var

        saver = tf.train.Saver(var_list=rv_dict)
        saver.restore(self.session, self.checkpoint_path)

        print("[.] variables initialized")

    def prepare_summaries(self):
        print("[.] preparing summaries")

        tf.summary.scalar("content loss", self.content_loss_v * self.content_weight)
        tf.summary.scalar("style loss", self.style_loss_v * self.style_weight)
        tf.summary.scalar("total variation loss", self.tv_loss_v * self.tv_weight)
        tf.summary.scalar("loss", self.loss_v)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.image("generated image", self.x[None])

        self.summaries = tf.summary.merge_all()
        self.summary_path = self.get_summary_path()
        self.prepare_summary_directory(self.summary_path)
        self.writer = tf.summary.FileWriter(self.summary_path, self.session.graph)

        print("[.] summaries prepared")

    def run_style_transfer(self):
        print("[.] running style transfer")

        for i in range(self.num_iterations + 1):
            print("[.] -> iteration", i)
            _, summaries_values, l, cl, sl, tvl = self.session.run([self.train_op, self.summaries, self.loss_v,
                                                                    self.content_loss_v, self.style_loss_v,
                                                                    self.tv_loss_v])

            self.writer.add_summary(summaries_values, i)
            print("[.]    loss: {}  content loss: {}  style loss: {}  tv loss: {}"
                  .format(l, cl, sl, tvl))

            if i % 10 == 0:
                print("[.] --> saving image")
                x = self.session.run([tf.squeeze(self.x)])
                x = image_utils.deprocess(x[0])
                image_utils.save_generated_image(x, self.summary_path, name=str(i))

        print("[.] style transfer done")

    def get_summary_path(self):
        path = os.path.join("summary", self.name)
        path = os.path.join(path, str(time.time()).split(".")[0])

        return path

    def prepare_summary_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "params.json"), "w") as file:
            params_dict = {
                "content_weight": self.content_weight,
                "style_weight": self.style_weight,
                "total_variation_weight": self.tv_weight,
                "learning_rate": self.learning_rate_value,
                "iterations_number": self.num_iterations,
                "content_layers": self.content_layers,
                "style_layers": self.style_layers
            }
            json.dump(params_dict, file)

    def get_content_features(self, image):
        content_features = self.get_features(image, self.content_layers)
        return content_features

    def get_style_features(self, image):
        style_features = self.get_features(image, self.style_layers)
        return style_features

    def get_features(self, image, layers):
        with tf.variable_scope("features", reuse=tf.AUTO_REUSE):
            with slim.arg_scope(self.model_arg_scope):
                _, endpoints = self.model(image[None], is_training=False, spatial_squeeze=False)

                features = []
                i = 0
                for layer in endpoints.keys():
                    if layer.endswith(layers[i]):
                        features.append(endpoints[layer])

                        i += 1
                        if i == len(layers):
                            break

        return features
