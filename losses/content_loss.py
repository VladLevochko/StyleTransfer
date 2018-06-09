import tensorflow as tf


class ContentLoss:
    @staticmethod
    def get_value(features_list_a, features_list_b):
        with tf.name_scope("content_loss_extraction"):
            with tf.variable_scope("content_loss_extraction"):
                loss = 0
                for features_a, features_b in zip(features_list_a, features_list_b):
                    loss += tf.reduce_sum((features_a - features_b) ** 2)

        return loss
