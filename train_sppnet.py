from spp_net import *

import os
import time

import numpy as np
from read_data import *

train_size = 3060
batch_size = 4
max_epochs =2
num_class = 102
eval_frequency = 100
max_steps = 10000

model_save_file = "./result/spp_pickleModel.ckpt"

def train():
    global_step = tf.Variable(0, trainable=False)
    spp_net = SPPnet('./data/vgg16.npy')
    spp_net.set_lr(0.0001, batch_size, train_size)
# load data
    print('load data')
    train_data, train_label,ishape = spp_net.input_data('data/resizedImage/','data/train.txt', batch_size, True)
    print("load done")
#    valid_data, valid_label, vshape = input_data_t('data/101_ObjectCategories/','data/valid.txt', 32)
    num_class = 102

# train
    print('train')
    logits = spp_net.inference(train_data, True, num_class)
    loss, accuracy = spp_net.loss(logits, train_label)
    opt, lr = spp_net.train(loss, global_step)
    print('train done')
# evaluation
#    eval_logits = spp_net.inference(valid_data, False, num_class)
#    eval_accuracy = spp_net.loss(eval_logits, valid_label)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()
        coord = tf.train.Coordinator()
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        start_time = time.time()
    #    print((FLAGS.max_epochs * train_size) // batch_size)
        for step in xrange(max_steps):
            _, loss_value, accu = sess.run([opt, loss, accuracy])
            if step % eval_frequency ==0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                    1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train accu: %.2f%%' % accu)         
        coord.request_stop()
        coord.join(threads)
'''
                if step % eval_frequency == 0:
                stop_time = time.time() - start_time
                start_time = time.time()
                print('epoch: %.2f , %.2f ms' % (step * batch_size /train_size,
                                                1000 * stop_time / eval_frequency)) 
                print('train loss: %.3f' % loss_value) 
                print('train error: %.2f%%' % accu)         
                eval_accu = sess.run([eval_accuracy])
                print('valid error: %.2f%%' % eval_accu)
                
                if((1- eval_accu) < best_error_rate):
                    if((1-eval_accu) < best_error_rate * 0.95):
                        if(patience<step *2):
                            patience = patience *2
                    best_error_rate = 1 - eval_accu 

            if step >= patience:
                saver.save(sess, model_save_file, global_step = step)
                break
            if (step +1) == (FLAGS.max_epochs * train_size) // batch_size:
                saver.save(sess, model_save_file, global_step = step) '''

if __name__ == '__main__':
    train()


