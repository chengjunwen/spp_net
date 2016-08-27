from __future__ import absolute_import                                                                         
from __future__ import division
from __future__ import print_function

import os
import logging
import math 
import sys
from spp_layer import SPPLayer
from read_data import *

import numpy as np
import tensorflow as tf
SEED = 1356
VGG_MEAN = [103.939, 116.779, 123.68]
stddev = 0.05
class SPPnet:
    def __init__(self,model_file=None):
        self.random_weight= False
        if model_file is None:
            self.random_weight = True
            logging.error('please inp ut model file')
        if not os.path.isfile(model_file):
            logging.error(('model file is not exist:'), model_file)
        self.wd = 5e-4
        self.stddev = 0.05
        self.param_dict = np.load(model_file).item()
        print('model file loaded')
    
    def input_data(self, data_dir, trainfile, batch_size, shuffle=True):
        return input_data_t(data_dir, trainfile, batch_size, shuffle)
    def _conv_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
            if shape ==None :
                filter = self.get_conv_filter(name)
                conv_bias = self.get_bias(name)
            else :
                initW = tf.truncated_normal_initializer(stddev = self.stddev)
                filter = tf.get_variable(name='filter', shape=shape, initializer=initW)
                
                initB = tf.constant_initializer(0.0)
                conv_bias = tf.get_variable(name='bias',shape=shape[3], initializer=initB)
            conv = tf.nn.conv2d(bottom, filter, strides=[1 ,1 ,1 ,1], padding='SAME')
            relu = tf.nn.relu( tf.nn.bias_add(conv, conv_bias) )
            
            return relu
    def _fc_layer(self, bottom, name, shape=None):
        with tf.variable_scope(name) as scope:
            if shape == None:
                weight = self.get_fc_weight(name)
                bias = self.get_bias(name)
            else:
                weight =self._variable_with_weight_decay(shape, self.stddev, self.wd)
                initB = tf.constant_initializer(0.0)
                bias = tf.get_variable(name='bias',shape=shape[1], initializer=initB)

            fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
            
            if name == 'fc8' :
                return fc
            else:
                relu = tf.nn.relu(fc)
                return relu

    def inference(self, data, train=True, num_class=1000):
        with tf.name_scope('Processing'):
            self.conv1_1 = self._conv_layer(data, 'conv1_1', [3,3,3,64])
            self.conv1_2 = self._conv_layer(self.conv1_1, 'conv1_2', [3,3,64,64])
            self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool1')

            self.conv2_1 = self._conv_layer(self.pool1, 'conv2_1', [3,3,64,128])
            self.conv2_2 = self._conv_layer(self.conv2_1, 'conv2_2', [3,3,128,128])
            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool2')

            self.conv3_1 = self._conv_layer(self.pool2, 'conv3_1', [3,3,128,256])
            self.conv3_2 = self._conv_layer(self.conv3_1, 'conv3_2', [3,3,256,256])
            self.conv3_3 = self._conv_layer(self.conv3_2, 'conv3_3', [3,3,256,256])
            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool3')

            self.conv4_1 = self._conv_layer(self.pool3, 'conv4_1', [3,3,256,512])
            self.conv4_2 = self._conv_layer(self.conv4_1, 'conv4_2', [3,3,512, 512])
            self.conv4_3 = self._conv_layer(self.conv4_2, 'conv4_3', [3,3,512,512])
            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME',name='pool4')
            
            self.conv5_1 = self._conv_layer(self.pool4, 'conv5_1', [3,3,512,512])
            self.conv5_2 = self._conv_layer(self.conv5_1, 'conv5_2', [3,3,512,512])
            self.conv5_3 = self._conv_layer(self.conv5_2, 'conv5_3', [3,3,512,512])
            
            bins = [3, 2, 1]
            map_size = self.conv5_3.get_shape().as_list()[2]
            print(self.conv5_3.get_shape())
            sppLayer = SPPLayer(bins, map_size)
            self.sppool = sppLayer.spatial_pyramid_pooling(self.conv5_3)
            
            numH = self.sppool.get_shape().as_list()[1]
            print(numH)
            self.fc6 = self._fc_layer(self.sppool, 'fc6', shape=[numH, 4096])
            if train:
                self.fc6 = tf.nn.dropout(self.fc6, 0.5, seed=SEED)
            
            self.fc7 = self._fc_layer(self.fc6, 'fc7',shape= [4096,4096])
            if train:
                self.fc7 = tf.nn.dropout(self.fc7, 0.5, seed=SEED)
            self.fc8 = self._fc_layer(self.fc7, 'fc8', shape=[4096,num_class])
            print('inference')
            return self.fc8
    
    def loss(self, logits, label=None):
            self.pred = tf.nn.softmax(logits)
            if label is not None:
                label = tf.cast(label, tf.int64)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits, label, name = 'cross_entropy_all')
                self.entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
#                tf.add_to_collection('losses', self.entropy_loss)
#                self.all_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
                
                correct_prediction = tf.equal(tf.argmax(logits,1), label)
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                return (self.entropy_loss, self.accuracy)
            else:
                return self.pred
    
    def train(self, loss, global_step):
        self.lr = tf.train.exponential_decay(self.lr, 
                global_step*self.batch_size, self.train_size*self.decay_epochs, 0.95, staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(loss,
                global_step = global_step)
        return (self.optimizer, self.lr)

    def set_lr(self, lr, batch_size, train_size, decay_epochs = 10):
        self.lr = lr
        self.batch_size = batch_size
        self.train_size = train_size
        self.decay_epochs = decay_epochs
    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.param_dict[name][0], dtype=tf.float32)
        shape = self.param_dict[name][0].shape
        print('conv Layer name: %s' % name)
        print('conv Layer shape: %s' % str(shape))
        var = tf.get_variable(name = 'filter', initializer=init, shape=shape)
#        if not tf.get_variable_scope().reuse:
#            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
#                                  name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)

        return var
    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.param_dict[name][0], dtype=tf.float32)
        shape = self.param_dict[name][0].shape
        print('fc Layer name: %s' % name)
        print('fc Layer shape: %s' % str(shape))
        var = tf.get_variable(name = 'weight', initializer=init, shape=shape)
#        if not tf.get_variable_scope().reuse:
#            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
#                                  name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)

        return var
    def get_bias(self,name):
        init = tf.constant_initializer(value=self.param_dict[name][1], dtype=tf.float32)
        shape = self.param_dict[name][1].shape
        var = tf.get_variable(name = 'bias', initializer=init, shape=shape)
        return var

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

#        if wd and (not tf.get_variable_scope().reuse):
#            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
#            tf.add_to_collection('losses', weight_decay)
        return var

