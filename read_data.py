#!/usr/bin/env python                                                                                

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

HEIGHT = 224
WIDTH = 224
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
def input_data(data_dir, trainfile, batch_size, shuffle=True):
    
    with open(trainfile) as fp:
        lines = fp.readlines()
    filenames = [line.rstrip('\n') for line in lines]
    filepaths = [ os.path.join(data_dir, name) for name in filenames ]
    filepaths = ops.convert_to_tensor(filepaths, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filepaths)
    image, label = read_celtach(filename_queue.dequeue())

    distored_image = tf.random_crop(image, [HEIGHT, WIDTH, 3])
    distorted_image = tf.image.random_flip_left_right(distored_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)


    return _generate_image_and_label_batch(float_image, label, min_queue_examples, batch_size, shuffle)

def read_celtach(filename_and_label_queue):

    filename, label = tf.decode_csv(filename_and_label_queue, [[""]], [[""]], " ")
    file_content = tf.read_file(filename)
    image = tf.image.decode_png(file_content)
    return (image, label)

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    print(label)

    return (images, tf.reshape(label_batch, [batch_size]), images.get_shape())


def read_labeled_image_list(data_dir, image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        line = os.path.join(data_dir, line)
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    print(filenames[0], filenames[1])
    return (filenames, labels)

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return (example, label)

def input_data_t(data_dir, trainfile, batch_size, shuffle=True):
    image_list, label_list = read_labeled_image_list(data_dir, trainfile)

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_list, dtype=dtypes.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=64,
                                            shuffle=True)

    image, label = read_images_from_disk(input_queue)

    distored_image = tf.random_crop(image, [HEIGHT, WIDTH, 3])
    distorted_image = tf.image.random_flip_left_right(distored_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)


    return _generate_image_and_label_batch(float_image, label, min_queue_examples, batch_size, shuffle)


