import tensorflow as tf
from solver import *

flags = tf.app.flags

#solver
flags.DEFINE_string("train_dir", "./models", "trained model save path")
flags.DEFINE_string("imgs_list_path", "./data/train5/", "images list file path")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_boolean("use_queue_loading", True, "whether to use_queue_loading")
flags.DEFINE_integer("num_epoch", 10, "train epoch num")
flags.DEFINE_integer("batch_size", 128, "batch_size")
flags.DEFINE_integer("img_size_0", 41, "img_size[0]")
flags.DEFINE_integer("img_size_1", 41, "img_size[1]")
flags.DEFINE_float("learning_rate", 4e-4, "learning rate")

conf = flags.FLAGS

def main(_):
  solver = Solver()
  solver.train()

if __name__ == '__main__':
    tf.app.run()