import tensorflow as tf
import os
import json
import numpy as np


class App:

    def __init__(self, model, model_helper, config, session):
        self._model = model
        self._model_helper = model_helper
        self._config = config
        self.session = session
        self._summary_directory = config['summary_directory']
        self._counter_file_name = 'counter'

    def get_content_loss(self, content_original, content_current):
        """
        Calculate content loss of content image and generated image.

        :param content_original: list of features of content image
        :param content_current: list of features of generated image
        :return: sum of squared elementwise differences of features
        """
        # with tf.variable_scope("content loss"):
        content_weights = self._config['content_weights']
        loss = 0
        layers_number = len(content_original)
        for i in range(layers_number):
            sum = tf.reduce_sum((content_original[i] - content_current[i]) ** 2)
            shape = content_original[i].shape
            # denom = shape[0] ** 2 * shape[1] * shape[2]
            denom = np.prod(shape)
            loss += content_weights[i] * sum / denom

        return loss

    def get_gram_matrix(self, features):
        """
        Compute the Gram matrix from features.

        Inputs:
        - features: Tensor of shape (1, H, W, C) giving features for
          a single image.
        - normalize: optional, whether to normalize the Gram matrix
            If True, divide the Gram matrix by the number of neurons (H * W * C)

        Returns:
        - gram: Tensor of shape (C, C) giving the (optionally normalized)
          Gram matrices for the input image.
        """
        shapes = tf.shape(features)
        reshaped_feautures = tf.reshape(features, [shapes[1] * shapes[2], shapes[3]])

        gram = tf.matmul(tf.transpose(reshaped_feautures), reshaped_feautures)
        # gram /= tf.cast(shapes[1] * shapes[2] * shapes[3], dtype=tf.float32)
        gram /= tf.cast(shapes[2] * shapes[3], dtype=tf.float32)

        return gram

    def get_style_loss(self, style_original, style_current):
        """
        Computes the style loss at a set of layers.

        Inputs:
        - feats: list of the features at every layer of the current image, as produced by
          the extract_features function.
        - style_layers: List of layer indices into feats giving the layers to include in the
          style loss.
        - style_targets: List of the same length as style_layers, where style_targets[i] is
          a Tensor giving the Gram matrix the source style image computed at
          layer style_layers[i].
        - style_weights: List of the same length as style_layers, where style_weights[i]
          is a scalar giving the weight for the style loss at layer style_layers[i].

        Returns:
        - style_loss: A Tensor contataining the scalar style loss.
        """
        style_weights = self._config['style_weights']
        loss = 0
        for i in range(len(style_original)):
            sum = tf.reduce_sum((style_original[i] - style_current[i]) ** 2)
            num_of_features = style_original[i].shape[0]
            denom = 4 * num_of_features ** 2
            loss += style_weights[i] * sum / denom

        return loss

    def get_tv_loss(self, img):
        """
        Compute total variation loss.

        Inputs:
        - img: Tensor of shape (1, H, W, 3) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.

        Returns:
        - loss: Tensor holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        tv_weight = self._config['tv_weight']
        w_variance = tf.reduce_sum((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)
        h_variance = tf.reduce_sum((img[:, 1:, :, :] - img[:, :-1, :, :]) ** 2)

        return tv_weight * (w_variance + h_variance)

    def get_content_features(self, content_image):
        """

        :param content_image:
        :return:
        """
        content_image = self._model_helper.preprocess_image(content_image)
        features = self._model.extract_features(self._model.image)
        content_target_vars = [features[i] for i in self._config['content_layers']]
        # with tf.Session() as sess:
        content_targets = self.session.run(content_target_vars,
                                           feed_dict={self._model.image: content_image})

        return content_targets

    def get_style_features(self, style_image):
        """

        :param style_image:
        :return:
        """
        style_image = self._model_helper.preprocess_image(style_image)
        features = self._model.extract_features(self._model.image)
        style_feature_vars = [features[i] for i in self._config['style_layers']]
        style_target_vars = []
        for style_feature_var in style_feature_vars:
            style_target_vars.append(self.get_gram_matrix(style_feature_var))
        # with tf.Session() as sess:
        style_targets = self.session.run(style_target_vars,
                                         feed_dict={self._model.image: style_image})

        return style_targets

    def get_features(self, image):
        """

        :param image:
        :return:
        """
        content_layers = self._config['content_layers']
        style_layers = self._config['style_layers']
        features = self._model.extract_features(image)
        content_target_vars = [features[i] for i in content_layers]
        style_feature_vars = [features[i] for i in style_layers]
        style_target_vars = []
        for style_feature_var in style_feature_vars:
            style_target_vars.append(self.get_gram_matrix(style_feature_var))

        return content_target_vars, style_target_vars

    def get_run_summary_path(self):
        with open(os.path.join(self._summary_directory, 'counter'), 'r') as counter_file:
            counter = int(counter_file.readline()) + 1
        with open(os.path.join(self._summary_directory, 'counter'), 'w') as counter_file:
            counter_file.write(str(counter))

        summary_path = os.path.join(self._summary_directory, 'run{}'.format(counter))
        return summary_path

    def store_hyperparameters(self, path):
        with open(os.path.join(path, 'hyperparameters.json'), 'w+') as f:
            json.dump(self._config, f)

    def style_transfer(self, content_image, style_image):
        """

        :param content_image: np_array with shape(width, height, c)
        :param style_image: np_array with shape(width, height, c)
        :return: np_array with shape(width, height, c) which represents content_image
                 styled appropriate to style_image
        """

        content_features = self.get_content_features(content_image[None])
        style_features = self.get_style_features(style_image[None])

        if self._config['init_random']:
            img_var = tf.Variable(tf.random_uniform(content_image.shape, 0, 1), name="image")
        else:
            img_var = tf.Variable(content_image, name="image")
        generated_content_features, generated_style_features = self.get_features(img_var[None])

        c_loss = self.get_content_loss(content_features, generated_content_features)
        s_loss = self.get_style_loss(style_features, generated_style_features)
        t_loss = self.get_tv_loss(img_var[None])
        loss = c_loss + s_loss + t_loss

        # Create and initialize the Adam optimizer
        lr_var = tf.Variable(self._config['learning_rate'], name="lr")
        # Create train_op that updates the generated image when run
        with tf.variable_scope("optimizer") as opt_scope:
            train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
        # Initialize the generated image and optimization variables
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
        # with tf.Session() as sess:
        self.session.run(tf.variables_initializer([lr_var, img_var] + opt_vars))

        # Create an op that will clamp the image values when run
        clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))

        # image = self._model_helper.deprocess_image(img_var[None].eval())

        tf.summary.scalar('content loss', c_loss)
        tf.summary.scalar('style loss', s_loss)
        tf.summary.scalar('total variance loss', t_loss)
        tf.summary.scalar('total loss', loss)
        tf.summary.scalar('learning rate', lr_var)
        # tf.summary.image("generated image", img_var[None])
        tf.summary.image("generated image", clamp_image_op[None])
        # tf.summary.image("generated image", image)

        merged = tf.summary.merge_all()
        run_summary_path = self.get_run_summary_path()
        writer = tf.summary.FileWriter(run_summary_path, self.session.graph)
        self.store_hyperparameters(run_summary_path)

        decay_at = self._config['decay_at']
        for t in range(self._config['num_iterations']):
            if t == decay_at:
                self.session.run(tf.assign(lr_var, self._config['decayed_lr']))
            _, summary, c_loss_v, s_loss_v, loss_v = self.session.run([train_op,
                                                                      merged, c_loss,
                                                                      s_loss, loss])
            writer.add_summary(summary, t)
            print("{} content loss {} style loss {} loss {}".format(t, c_loss_v, s_loss_v, loss_v))

        generated_image = tf.squeeze(img_var)
        generated_image = self.session.run(generated_image)
        # generated_image = self.session.run(img_var[None])

        return self._model_helper.deprocess_image(generated_image)

