import tensorflow as tf


class TotalVariationLoss:

    def get_value(self, tensor):
        with tf.name_scope("total_variation_loss_extraction"):
            shape = tf.shape(tensor)
            a = tensor[:shape[0] - 1, :shape[1] - 1, :] - tensor[1:, :shape[1] - 1, :]
            b = tensor[:shape[0] - 1, :shape[1] - 1, :] - tensor[:shape[0] - 1, 1:, :]

        return tf.reduce_sum(tf.pow((a - b), 1))  # TODO: set y to 1.25
