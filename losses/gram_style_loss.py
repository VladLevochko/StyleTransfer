import tensorflow as tf


class GramBasedStyleLoss:
    def get_value(self, features_list_a, features_list_b):
        with tf.variable_scope("style_loss"):
            loss = 0
            for features_a, features_b in zip(features_list_a, features_list_b):
                gm_a = self.get_gram_matrix(features_a)
                gm_b = self.get_gram_matrix(features_b)

                shape = tf.cast(tf.shape(features_a), dtype=tf.float32)

                loss += tf.reduce_sum((gm_a - gm_b) ** 2) / (4 * (shape[1] * shape[2]) ** 2 * shape[3] ** 2)

        return loss

    def get_gram_matrix(self, features):
        with tf.variable_scope("gram_matrix"):
            shape = tf.shape(features)
            reshaped_features = tf.reshape(features, [shape[1] * shape[2], shape[3]])

            gram = tf.matmul(reshaped_features, reshaped_features, transpose_a=True)

        return gram
