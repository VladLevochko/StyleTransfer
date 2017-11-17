import tensorflow as tf
import scipy.io
import numpy as np


class VGG19:

    def __init__(self, path_to_model, session):
        self._mean = np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        self.layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        self._weights, self._biases = self.get_weights_and_biases(path_to_model)
        self.image = tf.placeholder('float', shape=[None, None, None, 3], name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')

    def extract_features(self, input=None):
        layers = []
        x = input
        with tf.variable_scope("block1"):
            x = self.conv(x, self._weights[0], self._biases[0], 'conv1_1')
            layers.append(x)  # 0
            x = self.conv(x, self._weights[1], self._biases[1], 'conv1_2')
            layers.append(x)  # 1
            x = self.pool(x)
            layers.append(x)  # 2
        with tf.variable_scope("block2"):
            x = self.conv(x, self._weights[2], self._biases[2], 'conv2_1')
            layers.append(x)  # 3
            x = self.conv(x, self._weights[3], self._biases[3], 'conv2_2')
            layers.append(x)  # 4
            x = self.pool(x)
            layers.append(x)  # 5
        with tf.variable_scope("block3"):
            x = self.conv(x, self._weights[4], self._biases[4], 'conv3_1')
            layers.append(x)  # 6
            x = self.conv(x, self._weights[5], self._biases[5], 'conv3_2')
            layers.append(x)  # 7
            x = self.conv(x, self._weights[6], self._biases[6], 'conv3_3')
            layers.append(x)  # 8
            x = self.conv(x, self._weights[7], self._biases[7], 'conv3_4')
            layers.append(x)  # 9
            x = self.pool(x)
            layers.append(x)  # 10
        with tf.variable_scope("block4"):
            x = self.conv(x, self._weights[8], self._biases[8], 'conv4_1')
            layers.append(x)  # 11
            x = self.conv(x, self._weights[9], self._biases[9], 'conv4_2')
            layers.append(x)  # 12
            x = self.conv(x, self._weights[10], self._biases[10], 'conv4_3')
            layers.append(x)  # 13
            x = self.conv(x, self._weights[11], self._biases[11], 'conv4_4')
            layers.append(x)  # 14
            x = self.pool(x)
            layers.append(x)  # 15
        with tf.variable_scope("block5"):
            x = self.conv(x, self._weights[12], self._biases[12], 'conv5_1')
            layers.append(x)  # 16
            x = self.conv(x, self._weights[13], self._biases[13], 'conv5_2')
            layers.append(x)  # 17
            x = self.conv(x, self._weights[14], self._biases[14], 'conv5_3')
            layers.append(x)  # 18
            x = self.conv(x, self._weights[15], self._biases[15], 'conv5_4')
            layers.append(x)  # 19
            x = self.pool(x)
            layers.append(x)  # 20

        return layers



    def build_net(self, input):
        net = {'input': input - self._mean}
        net['conv1_1'] = self.conv(net['input'], self._weights[0], self._biases[0], 'conv1_1')
        net['conv1_2'] = self.conv(net['conv1_1'], self._weights[2], self._biases[2], 'conv1_1')
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv(net['pool1'], self._weights[5], self._biases[5], 'conv1_1')
        net['conv2_2'] = self.conv(net['conv2_1'], self._weights[7], self._biases[7], 'conv1_1')
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv(net['pool2'], self._weights[10], self._biases[10], 'conv1_1')
        net['conv3_2'] = self.conv(net['conv3_1'], self._weights[12], self._biases[12], 'conv1_1')
        net['conv3_3'] = self.conv(net['conv3_2'], self._weights[14], self._biases[14], 'conv1_1')
        net['conv3_4'] = self.conv(net['conv3_3'], self._weights[16], self._biases[16], 'conv1_1')
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv(net['pool3'], self._weights[19], self._biases[19], 'conv1_1')
        net['conv4_2'] = self.conv(net['conv4_1'], self._weights[21], self._biases[21], 'conv1_1')
        net['conv4_3'] = self.conv(net['conv4_2'], self._weights[23], self._biases[23], 'conv1_1')
        net['conv4_4'] = self.conv(net['conv4_3'], self._weights[25], self._biases[25], 'conv1_1')
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv(net['pool4'], self._weights[28], self._biases[28], 'conv1_1')
        net['conv5_2'] = self.conv(net['conv5_1'], self._weights[30], self._biases[30], 'conv1_1')
        net['conv5_3'] = self.conv(net['conv5_2'], self._weights[32], self._biases[32], 'conv1_1')
        net['conv5_4'] = self.conv(net['conv5_3'], self._weights[34], self._biases[34], 'conv1_1')
        net['pool5'] = self.pool(net['conv5_4'])

        return net

    def conv(self, input, weights, biases, name=None):
        convolved = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME', name=name)
        rectified = tf.nn.relu(convolved + biases)

        return rectified

    def pool(self, input):
        return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_weights_and_biases(self, path_to_model):
        vgg_rawnet = scipy.io.loadmat(path_to_model)
        vgg_layers = vgg_rawnet['layers'][0]

        # weights = np.zeros(len(vgg_layers))
        # biases = np.zeros(len(vgg_layers))

        weights = []
        biases = []

        for layer in self.layers:
            weight = vgg_layers[layer][0][0][2][0][0]
            # weights[i] = tf.constant(weight)
            weights.append(tf.constant(weight))
            bias = vgg_layers[layer][0][0][2][0][1]
            # biases[i] = tf.constant(np.reshape(bias, bias.size))  # what is going on here?
            biases.append(tf.constant(np.reshape(bias, (bias.size,))))

        return weights, biases
