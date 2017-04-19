from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io
import threading
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
        self.use_queue_loading = conf.use_queue_loading
        self.dataset = DataSet(conf.imgs_list_path, self.num_epoch, self.batch_size)

        if self.use_queue_loading:
            self.threads = []
            self.num_thread = 10
            self.queue_lr = tf.placeholder(tf.float32, shape=(self.img_size_0, self.img_size_1, 1))
            self.queue_hr = tf.placeholder(tf.float32, shape=(self.img_size_0, self.img_size_1, 1))
            queue = tf.FIFOQueue(10000, [tf.float32, tf.float32],
                                 [[self.img_size_0, self.img_size_1, 1], [self.img_size_0, self.img_size_0, 1]])
            self.enqueue_op = queue.enqueue([self.queue_lr, self.queue_hr])
            train_input, train_gt = queue.dequeue_many(self.batch_size)

        if conf.use_gpu:
            device_str = '/gpu:0'
        else:
            device_str = '/cpu:0'
        with tf.device(device_str):
            if self.use_queue_loading:
                self.net = Net(train_input, train_gt, 'cnn_sr')
            else:
                self.train_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size_0, self.img_size_1, 1))
                self.train_gt = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size_0, self.img_size_1, 1))
                self.net = Net(self.train_input, self.train_gt, 'cnn_sr')

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

        if self.use_queue_loading:
            coord = tf.train.Coordinator()
            self.__start_queue(sess, coord, self.enqueue_op, self.num_thread)
            try:
                for epoch in range(0, self.num_epoch):
                    for step in range(self.dataset.total_sample // self.batch_size):
                        if coord.should_stop():
                            break
                        _, l = sess.run([self.train_op, self.net.loss])
                        print("[epoch %2.4f] loss %.4f\t " %
                              (epoch + (float(step) * self.batch_size / self.dataset.total_sample),
                               np.sum(l) / self.batch_size))
                    saver.save(sess, checkpoint_path, global_step=self.global_step)
            finally:
                coord.request_stop()
                coord.join(self.threads)
        else:
            for epoch in range(0, self.num_epoch):
                for step in range(self.dataset.total_sample // self.batch_size):
                    offset = step * self.batch_size
                    input_data, gt_data = self.dataset.get_image_batch(offset, self.batch_size,
                                                                       self.img_size_0, self.img_size_1)
                    _, l = sess.run([self.train_op, self.net.loss],
                                    feed_dict={self.train_input: input_data, self.train_gt: gt_data})
                    print("[epoch %2.4f] loss %.4f\t " %
                          (epoch + (float(step) * self.batch_size / self.dataset.total_sample),
                           np.sum(l) / self.batch_size))

                saver.save(sess, checkpoint_path, global_step=self.global_step)

        print('Done training ------------')


    # queue_loading
    def __start_queue(self, sess, coord, enqueue_op, num_thread):

        for i in range(num_thread):
            length = len(self.dataset.image_list) // num_thread
            t = threading.Thread(target=self.__load_and_enqueue,
                                 args=(sess, coord, self.dataset.image_list[i * length:(i + 1) * length],
                                       enqueue_op, i, num_thread))
            self.threads.append(t)
            t.start()

    def __load_and_enqueue(self, sess, coord, image_list, enqueue_op, idx=0, num_thread=1):
        count = 0
        length = len(image_list)
        try:
            while not coord.should_stop():
                i = count % length
                # i = random.randint(0, length-1)
                input_img = scipy.io.loadmat(image_list[i][1])['patch'].reshape([self.img_size_0, self.img_size_1, 1])
                gt_img = scipy.io.loadmat(image_list[i][0])['patch'].reshape([self.img_size_0, self.img_size_1, 1])
                sess.run(enqueue_op, feed_dict={self.queue_lr: input_img, self.queue_hr: gt_img})
                count += 1
                #if count % 100 == 0:
                    #print("[thread:", idx, "]", "enqueue...", count)
        except Exception as e:
            print("stopping...", idx, e)


