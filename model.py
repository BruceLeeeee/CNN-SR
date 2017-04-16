from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Net(object):
    def __init__(self, hr_images, lr_images, scope):
        with tf.variable_scope(scope) as scope:
            self.construct_net(hr_images, lr_images)

    def construct_net(self, hr_images, lr_images):

        input_tensor = lr_images
        self.weights = []
        tensor = None

        conv_00_w = tf.get_variable("conv_00_w", [3, 3, 1, 64], tf.float32,
                                    tf.truncated_normal_initializer(stddev=0.1))
        conv_00_b = tf.get_variable("conv_00_b", [64], tf.float32, tf.constant_initializer(0.0))
        self.weights.append(conv_00_w)
        self.weights.append(conv_00_b)
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'),
                                             conv_00_b))

        for i in range(18):
            conv_w = tf.get_variable("conv_%02d_w" % (i + 1), [3, 3, 64, 64], tf.float32,
                                       tf.truncated_normal_initializer(stddev=0.1))
            conv_b = tf.get_variable("conv_%02d_b" % (i + 1), [64], tf.float32, tf.constant_initializer(0.0))
            self.weights.append(conv_w)
            self.weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'),
                                                 conv_b))

        conv_w = tf.get_variable("conv_20_w", [3, 3, 64, 1], tf.float32,
                                tf.truncated_normal_initializer(stddev=0.1))
        conv_b = tf.get_variable("conv_20_b", [1], tf.float32, tf.constant_initializer(0.0))
        self.weights.append(conv_w)
        self.weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

        self.output_tensor = tf.add(tensor, input_tensor)

        self.loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.output_tensor, hr_images)))

        #tf.summary.scalar('loss', self.loss)








