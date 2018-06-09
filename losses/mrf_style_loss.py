import tensorflow as tf


class MrfBasedStyleLoss:
    def get_value(self, source_features_list, combination_features_list, patch_size=3, stride=1):
        with tf.name_scope("style_loss_extraction"):
            with tf.variable_scope("style_loss_extraction"):
                loss = 0.
                for source_features, combination_features in zip(source_features_list, combination_features_list):

                    combination_patches, combination_patches_norm = self.make_patches(combination_features, patch_size, stride)
                    source_patches, source_patches_norm = self.make_patches(source_features, patch_size, stride)

                    patch_ids = self.find_patch_matches(combination_patches, combination_patches_norm,
                                                        tf.divide(source_patches, source_patches_norm))
                    best_source_patches = tf.gather(source_patches, patch_ids)
                    loss += tf.reduce_sum((best_source_patches - combination_patches) ** 2)

        return loss / patch_size ** 2

    def make_patches(self, x, patch_size, stride):
        with tf.name_scope("patch_making"):
            with tf.variable_scope("make_patches"):
                ksizes = [1, patch_size, patch_size, 1]
                strides = [1, stride, stride, 1]
                rates = [1] * 4
                patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="VALID")

                patches = tf.reshape(patches, (-1, x.shape[3], patch_size, patch_size))
                patches_norm = tf.sqrt(tf.reduce_sum(patches ** 2, axis=(1, 2, 3), keep_dims=True))

        return patches, patches_norm

    def find_patch_matches(self, a, a_norm, b):
        with tf.name_scope("patch_searching"):
            with tf.variable_scope("find_patch_matches"):
                a = tf.transpose(a, [0, 2, 3, 1])
                filter = tf.transpose(b, [2, 3, 1, 0])
                convs = tf.nn.conv2d(a, filter, [1] * 4, padding="VALID")
                argmax = tf.argmax(tf.transpose(convs / a_norm, [0, 3, 1, 2]), axis=1)
                argmax = tf.squeeze(argmax)

        return argmax
