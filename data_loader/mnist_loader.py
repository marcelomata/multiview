import pickle

from tqdm import tqdm

import tensorflow as tf
import numpy as np

class MnistDataLoaderNumpy:
    """
    This will contain the dataset api
    It will load the numpy files from the pkl file which is dumped by prepare_mnist.py script
    Please make sure that you have included all of the needed config
    Thanks..
    """

    def __init__(self, config):
        self.config = config

        with open(self.config.data_numpy_pkl, "rb") as f:
            self.data_pkl = pickle.load(f)

        self.x_train = self.data_pkl['x_train']
        self.y_train = self.data_pkl['y_train']
        self.x_validate = self.data_pkl['x_validate']
        self.y_validate = self.data_pkl['y_validate']
        self.x_test = self.data_pkl['x_test']
        self.y_test = self.data_pkl['y_test']

        print('x_train: ', self.x_train.shape, self.x_train.dtype)
        print('y_train: ', self.y_train.shape, self.y_train.dtype)
        print('x_validate: ', self.x_validate.shape, self.x_train.dtype)
        print('y_validate: ', self.y_validate.shape, self.y_validate.dtype)
        print('x_test: ', self.x_test.shape, self.x_test.dtype)
        print('y_test: ', self.y_test.shape, self.y_test.dtype)

        self.train_len = self.x_train.shape[0]
        self.validate_len = self.x_validate.shape[0]
        self.test_len = self.x_test.shape[0]

        self.num_iterations_train = (self.train_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_validate = (self.validate_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test = (self.test_len + self.config.batch_size - 1) // self.config.batch_size

        print("Data loaded successfully..")

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self.build_dataset_api()

    def build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.float32, [None] + list(self.x_train.shape[1:]))
            self.labels_placeholder = tf.placeholder(tf.int64, [None, ])

            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            self.dataset = self.dataset.repeat().batch(self.config.batch_size)

            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)

            self.init_iterator_op = self.iterator.make_initializer(self.dataset)

            self.next_batch = self.iterator.get_next()

            print("X_batch shape dtype: ", self.next_batch[0].shape)
            print("Y_batch shape dtype: ", self.next_batch[1].shape)

    def initialize(self, sess, state):
        if state == 'train':
            idx = np.random.choice(self.train_len, self.train_len, replace=False)
            self.x_train = self.x_train[idx]
            self.y_train = self.y_train[idx]
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_train,
                                                       self.labels_placeholder: self.y_train})
        elif state == 'validate':
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_validate,
                                                       self.labels_placeholder: self.y_validate})

        else:
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_test,
                                                       self.labels_placeholder: self.y_test})

    def get_input(self):
        return self.next_batch


def main():
    class Config:
        data_numpy_pkl = "../data/mnist/data_numpy.pkl"
        data_mode = "numpy"

        image_height = 28
        image_width = 28
        batch_size = 8

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = Cifar100DataLoaderNumpy(Config)

    x, y = data_loader.next_batch

    data_loader.initialize(sess, state='train')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, state='validate')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, state='test')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

if __name__ == '__main__':
    main()
