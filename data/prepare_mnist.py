"""
This file is for preparing mnist and extracting it from the binary files

# Please first download cifar100 dataset and extract it in data folder here!!
# Then run this script to prepare the data of cifar100

- Generates numpys
- Generates images
- Generates tfrecords
"""
import os

import gzip 
import numpy as np
import imageio
import _pickle as cPickle
import pickle
from tqdm import tqdm

import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def save_imgs_to_disk(path, arr, file_names):
    for i, img in tqdm(enumerate(arr)):
        imageio.imwrite(path + file_names[i], img, 'PNG-PIL')


def save_numpy_to_disk(path, arr):
    np.save(path, arr)


def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())


def main():
    f = gzip.open('./mnist/mnist.pkl.gz', 'rb')
    list_train, list_validate, list_test = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    
    x_train, y_train = list_train
    x_validate, y_validate = list_validate
    x_test, y_test = list_test

    x_train = np.transpose(x_train.reshape((-1, 1, 28, 28)), (0, 2, 3, 1))
    x_validate = np.transpose(x_validate.reshape((-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = np.transpose(x_test.reshape((-1, 1, 28, 28)), (0, 2, 3, 1))

    print(x_train.shape)
    print(x_train.dtype)
    print(y_train.shape)
    print(y_train.dtype)
    print(x_validate.shape)
    print(x_validate.dtype)
    print(y_validate.shape)
    print(y_validate.dtype)
    print(x_test.shape)
    print(x_test.dtype)
    print(y_test.shape)
    print(y_test.dtype)

    print("Saving the file name of imgs NOT IMPLEMENTED")

    print("Saving the imgs NOT IMPLEMENTED")

    print("Saving the numpys to the disk..")

    save_numpy_to_disk('mnist/x_train.npy', x_train)
    save_numpy_to_disk('mnist/y_train.npy', y_train)
    save_numpy_to_disk('mnist/x_validate.npy', x_validate)
    save_numpy_to_disk('mnist/y_validate.npy', y_validate)
    save_numpy_to_disk('mnist/x_test.npy', x_test)
    save_numpy_to_disk('mnist/y_test.npy', y_test)

    print("Numpys saved successfully")

    print("Saving the data numpy pickle to the disk..")

    # SAVE ALL the data with one pickle
    with open('mnist/data_numpy.pkl', 'wb')as f:
        pickle.dump({'x_train': x_train,
                     'y_train': y_train,
                     'x_validate': x_validate,
                     'y_validate': y_validate,
                     'x_test': x_test,
                     'y_test': y_test,
                     }, f)

    print("DATA NUMPY PICKLE saved successfully..")

    print('saving tfrecord..')

    save_tfrecord_to_disk('mnist/train.tfrecord', x_train, y_train)
    save_tfrecord_to_disk('mnist/validate.tfrecord', x_validate, y_validate)  
    save_tfrecord_to_disk('mnist/test.tfrecord', x_test, y_test)

    print('tfrecord saved successfully..')

if __name__ == '__main__':
    main()
