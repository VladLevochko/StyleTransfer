import tensorflow as tf


class MrfBasedStyleLoss:
    def get_value(self, features_list_a, features_list_b, patch_size=3, stride=1):
        with tf.variable_scope("style_loss"):
            loss = 0.
            for features_a, features_b in zip(features_list_a, features_list_b):

                loss = tf.Print(loss, [features_a.shape, features_b.shape])

                patches_b, patches_norm_b = self.make_patches(features_b, patch_size, stride)
                patches_a, patches_norm_a = self.make_patches(features_a, patch_size, stride)

                patch_ids = self.find_patch_matches(patches_b, patches_norm_b, tf.divide(patches_a, patches_norm_a))
                best_a_patches = tf.gather(patches_a, patch_ids)
                loss += tf.reduce_sum((best_a_patches - patches_b) ** 2 / patch_size ** 2)

        return loss

    def make_patches(self, x, patch_size, stride):
        with tf.variable_scope("make_patches"):
            ksizes = [1, patch_size, patch_size, 1]
            strides = [1, stride, stride, 1]
            rates = [1] * 4
            patches = tf.extract_image_patches(x, ksizes, strides, rates, padding="VALID")

            # patches = tf.Print(patches, [x.shape, patches.shape], message="x.shape, p.shape ")

            patches = tf.reshape(patches, (x.shape[3], -1, patch_size, patch_size))
            patches = tf.transpose(patches, [1, 0, 2, 3])
            patches_norm = tf.sqrt(tf.reduce_sum(patches ** 2, axis=(1, 2, 3), keep_dims=True))

        return patches, patches_norm

    def find_patch_matches(self, a, a_norm, b):
        with tf.variable_scope("find_patch_matches"):
            a = tf.transpose(a, [0, 2, 3, 1])
            filter = tf.transpose(b, [2, 3, 1, 0])
            convs = tf.nn.conv2d(a, filter, [1] * 4, padding="VALID")
            argmax = tf.argmax(tf.transpose(convs / a_norm, [0, 3, 1, 2]), axis=1)
            argmax = tf.squeeze(argmax)

        return argmax
