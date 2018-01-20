import os
import tensorflow as tf
import numpy as np


class StyleTransfer:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.style_layers = config["style_layers"]
        self.content_layers = config["content_layers"]
        self.style_weights = config["style_weights"]
        self.style_loss_weight = config["style_loss_weight"]
        self.content_loss_weight = config["content_loss_weight"]
        self.learning_rate = tf.Variable(config["learning_rate"], name="learning_rate")
        self.num_iterations = config["num_iterations"]
        self.summary_directory = config["summary_directory"]

    def get_content_loss(self, original_content, generated_content):
        with tf.variable_scope("content_loss"):
            loss = 0
            for i in range(len(original_content)):
                sum = tf.reduce_sum((original_content[i] - generated_content[i]) ** 2)
                loss += sum

        return loss / 2

    def get_content_features(self, image, reuse=None):
        with tf.variable_scope("features_extraction", reuse=reuse):
            _, endpoints = self.model(image[None], spatial_squeeze=False)

            content_features = []
            i = 0
            for layer in endpoints.keys():
                if layer.endswith(self.content_layers[i]):
                    content_features.append(endpoints[layer])

                    i += 1
                    if i == len(self.content_layers):
                        break

        return content_features

    def get_style_loss(self, original_style, generated_style):
        with tf.variable_scope("style_loss"):
            loss = 0
            for i in range(len(original_style)):
                sum = tf.reduce_sum((original_style[i] - generated_style[i]) ** 2)
                loss += self.style_weights[i] * sum

        return loss

    def get_gram_matrix(self, features):
        with tf.variable_scope("gram_matrix"):
            shapes = tf.shape(features)
            reshaped_features = tf.reshape(features, [shapes[1] * shapes[2], shapes[3]])

            gram = tf.matmul(tf.transpose(reshaped_features), reshaped_features)
            gram /= tf.cast(2 * shapes[1] * shapes[2] * shapes[3], dtype=tf.float32)

        return gram

    def get_style_features(self, image, reuse=None):
        with tf.variable_scope("feature_extraction", reuse=reuse):
            _, endpoints = self.model(image[None], spatial_squeeze=False)
            style_features = []

            i = 0tensor
            for layer in endpoints.keys():
                if layer.endswith(self.style_layers[i]):
                    style_features.append(endpoints[layer])

                    i += 1
                    if i == len(self.style_layers):
                        break

        return style_features

    def transfer(self, content_image, style_image):
        # add to saver ops from model
        with tf.variable_scope(self.config["model_type"]) as scope:
            operations = scope.
        saver = tf.train.Saver()

        with tf.Session() as session:
            saver.restore(session, self.config["checkpoint"])
            # tf.train.load_checkpoint(self.config["checkpoint"])

            preprocessed_content_image = self.preprocess(content_image)
            preprocessed_style_image = self.preprocess(style_image)

            content_features = self.get_content_features(preprocessed_content_image)
            style_features = self.get_style_features(preprocessed_style_image)

            original_content_features, original_style_features = session.run([content_features, style_features])

            generated_image = tf.Variable(tf.random_uniform(content_image.shape, 0, 1))
            generated_content_features = self.get_content_features(generated_image, reuse=True)
            generated_style_features = self.get_style_features(generated_image, reuse=True)

            content_loss = self.get_content_loss(original_content_features,
                                                 generated_content_features)
            content_loss = tf.scalar_mul(self.content_loss_weight, content_loss)
            style_loss = self.get_style_loss(original_style_features,
                                             generated_style_features)
            style_loss = tf.scalar_mul(self.style_loss_weight, style_loss)

            loss = content_loss + style_loss

            with tf.variable_scope("optimizer") as opt_scope:
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=[generated_image])
            opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)

            tf.summary.scalar("content loss", content_loss)
            tf.summary.scalar("style loss", style_loss)
            tf.summary.scalar("total loss", loss)
            tf.summary.image("generated image", generated_image)
            summaries = tf.summary.merge_all()

            summary_path = self.get_summary_path()

            session.run(tf.variables_initializer([self.learning_rate, generated_image] + opt_vars))
            writer = tf.summary.FileWriter(summary_path, session.graph)

            for i in range(self.num_iterations):
                _, summary, content_loss_value, \
                    style_loss_value, total_loss_value = session.run(
                        [train_op, summaries, content_loss, style_loss, loss]
                    )

                writer.add_summary(summary, i)
                print("{} content loss {} style loss {} loss {}".format(
                    i, content_loss_value, style_loss_value, total_loss_value))

            generated_image = tf.squeeze(generated_image)
            generated_image = session.run(generated_image)

        return self.deprocess(generated_image)

    def get_summary_path(self):
        with open(os.path.join(self.summary_directory, 'counter'), 'r') as counter_file:
            counter = int(counter_file.readline()) + 1
        with open(os.path.join(self.summary_directory, 'counter'), 'w') as counter_file:
            counter_file.write(str(counter))

        summary_path = os.path.join(self.summary_directory, 'run{}'.format(counter))
        return summary_path

    def preprocess(self, image):
        return (image.astype(np.float32) / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def deprocess(self, image):
        image = (image * self.IMAGENET_STD + self.IMAGENET_MEAN)
        return np.clip(255 * image, 0.0, 255.0).astype(np.uint8)
