from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from data import *
from model import *

import os
import time

flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
    def __init__(self):
        self.train_dir = conf.train_dir
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        self.num_epoch = conf.num_epoch
        self.batch_size = conf.batch_size
        self.img_size_0 = conf.img_size_0
        self.img_size_1 = conf.img_size_1
        self.learning_rate = conf.learning_rate
        self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size)

        if conf.use_gpu:
            device_str = '/gpu:0'
        else:
            device_str = '/cpu:0'
        with tf.device(device_str):
            self.train_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size_1, self.img_size_0, 1))
            self.train_gt = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size_1, self.img_size_0, 1))
            self.net = Net(self.train_gt, self.train_input, 'cnn_sr')
            # optimizer
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                       500000, 0.5, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)

    def train(self):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(init_op)

        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
        for epoch in range(0, self.num_epoch):
            for step in range(self.dataset.total_sample // self.batch_size):
                offset = step * self.batch_size
                input_data, gt_data = self.dataset.get_image_batch(offset, self.batch_size,
                                                                   self.img_size_0, self.img_size_1)
                _, l = sess.run([self.train_op, self.net.loss],
                                feed_dict={self.train_input: input_data, self.train_gt: gt_data})
                print("[epoch %2.4f] loss %.4f\t " % (
                    epoch + (float(step) * self.batch_size / self.dataset.total_sample), np.sum(l) / self.batch_size))

            saver.save(sess, checkpoint_path, global_step=self.global_step)

        print('Done training ------------')