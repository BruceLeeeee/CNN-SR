from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Net(object):
    def __init__(self, hr_images, lr_images, scope):
        self.weights = []
        self.biases = []
        with tf.variable_scope(scope):
            self.__construct_net(hr_images, lr_images)

    def __construct_net(self, hr_images, lr_images):
        input_tensor = lr_images
        self.conv1 = self.__conv_layer(input_tensor, filters_shape=[3, 3, 1, 64], name='conv1')
        self.conv2 = self.__conv_layer(self.conv1, filters_shape=[3, 3, 64, 64], name='conv2')
        self.conv3 = self.__conv_layer(self.conv2, filters_shape=[3, 3, 64, 64], name='conv3')
        self.conv4 = self.__conv_layer(self.conv3, filters_shape=[3, 3, 64, 64], name='conv4')
        self.conv5 = self.__conv_layer(self.conv4, filters_shape=[3, 3, 64, 64], name='conv5')
        self.conv6 = self.__conv_layer(self.conv5, filters_shape=[3, 3, 64, 64], name='conv6')
        self.conv7 = self.__conv_layer(self.conv6, filters_shape=[3, 3, 64, 64], name='conv7')
        self.conv8 = self.__conv_layer(self.conv7, filters_shape=[3, 3, 64, 64], name='conv8')
        self.conv9 = self.__conv_layer(self.conv8, filters_shape=[3, 3, 64, 64], name='conv9')
        self.conv10 = self.__conv_layer(self.conv9, filters_shape=[3, 3, 64, 64], name='conv10')
        self.conv11 = self.__conv_layer(self.conv10, filters_shape=[3, 3, 64, 64], name='conv11')
        self.conv12 = self.__conv_layer(self.conv11, filters_shape=[3, 3, 64, 64], name='conv12')
        self.conv13 = self.__conv_layer(self.conv12, filters_shape=[3, 3, 64, 64], name='conv13')
        self.conv14 = self.__conv_layer(self.conv13, filters_shape=[3, 3, 64, 64], name='conv14')
        self.conv15 = self.__conv_layer(self.conv14, filters_shape=[3, 3, 64, 64], name='conv15')
        self.conv16 = self.__conv_layer(self.conv15, filters_shape=[3, 3, 64, 64], name='conv16')
        self.conv17 = self.__conv_layer(self.conv16, filters_shape=[3, 3, 64, 64], name='conv17')
        self.conv18 = self.__conv_layer(self.conv17, filters_shape=[3, 3, 64, 64], name='conv18')
        self.conv19 = self.__conv_layer(self.conv18, filters_shape=[3, 3, 64, 64], name='conv19')
        self.conv20 = self.__conv_layer(self.conv19, filters_shape=[3, 3, 64, 1], name='conv20')
        self.output_tensor = tf.add(self.conv20, input_tensor)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.output_tensor, hr_images)))
        for w in self.weights:
            self.loss += tf.nn.l2_loss(w) * 1e-4

    def __get_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0., stddev=.1), dtype=tf.float32)

    def __conv_layer(self, inputs, filters_shape, name, activation=tf.nn.relu):
        with tf.variable_scope(name):
            if name == 'conv20':
                activation = None

            filters = self.__get_weights(filters_shape)
            self.weights.append(filters)
            filter_maps = tf.nn.conv2d(inputs, filters, [1, 1, 1, 1], padding='SAME')
            num_out_maps = filters_shape[3]
            bias = tf.Variable(tf.constant(.0, shape=[num_out_maps]))
            self.biases.append(bias)
            filter_maps = tf.nn.bias_add(filter_maps, bias)

            if activation:
                return activation(filter_maps)
            else:
                return filter_maps