import tensorflow as tf


# a = tf.constant([[1, 2, 3], [2, 3, 4]], name="a")
# b = tf.constant([[2, 5, 4], [7, 4, 5]], name="b")
#
# op = tf.reduce_sum((b - a) ** 2)
#
# with tf.Session() as session:
#     print(session.run(op))
#
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file("checkpoints/vgg_19.ckpt", tensor_name='', all_tensors=True)

a = 2
b = 3
c = 4

print((a * b) ** 2 * c ** 2)
print(((a * b) ** 2) * (c ** 2))
