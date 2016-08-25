#!/usr/bin/env python                                                                                

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import sklearn.utils import shuffle
import tensorflow as tf

from tensorflow.python.framework import ops

HEIGHT = 224
WEIGHT = 224

def input_data(data_dir, trainfile, batch_size, shuffle=True):
    
    with open(trainfile) as fp:
        lines = fp.readlines()
    filenames = [line.rstrip('\n') for line in lines]
    filepaths = [ os.path.join(data_dir, name) for name in filenames ]
    filename_queue = tf.train.string_input_producer(filepaths)
    image, label = read_celtach(filename_queue.dequeue())

    distored_image = tf.random_crop(image, [HEIGHT, WIDTH, 3])
    distorted_image = tf.image.random_flip_left_right(dostored_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_constrat(distorted_image,lower=0.2, upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)


    return _generate_image_and_label_batch(float_image, label, min_queue_examples, batch_size, shuffle)

def read_celtach(filename_and_label_queue):

    filename, label = tf.decode_csv(filename_and_label_tensor, [[""]], [[""]], " ")
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

    return images, tf.reshape(label_batch, [batch_size])
