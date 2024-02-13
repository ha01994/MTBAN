import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random, time, sys, os, csv
from utils import *
import tensorflow as tf
import numpy as np
from model import TemporalConvNet
import scipy
from scipy.special import softmax


jobId = sys.argv[1]
accession = sys.argv[2]



if True:
    data_aa = np.load('_job%s/data_aa.npy'%jobId) #(num_seqs, seq_len)
    wt_seq = np.reshape(data_aa[0,:], (1,-1))
    new_weights = np.load('_job%s/weights.npy'%jobId)
    

    aa_size = 24
    seq_len = int(np.shape(data_aa)[1])
    keep_prob = 0.7
    batch_size = 128
    training_steps = 200000
    nhid = 128
    k, n = return_k_n(seq_len)

    train_data, train_w = data_aa, new_weights

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(10)
        X = tf.placeholder(tf.int32, [None, seq_len])
        W = tf.placeholder(tf.float32, [None])
        p_keep = tf.placeholder('float')
        learning_rate = tf.placeholder('float')

        out, wa_s = TemporalConvNet(aa_size, seq_len, [nhid]*n, k, p_keep)(X)

        loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                              labels=X, logits=out), axis=1), (W/tf.reduce_mean(W))))

        test_out = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X, logits=out), axis=1)

        global_step = tf.Variable(0, trainable=False)

        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)    

        init = tf.global_variables_initializer()

        print("Trainable parameters:")
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        print('starting training (teacher)...')
        sess.run(init)
        stored_exception=None

        for step in range(training_steps):
            X_b, W_b = next_batch(train_data, train_w, batch_size) 
            lr = 0.001
            sess.run(train_op, feed_dict={X: X_b, W: W_b, p_keep: keep_prob, learning_rate: lr})

            if step % 10000 == 0:
                print('step: %d'%step)            

            if step == training_steps - 1:           
                for i in range(0, int(len(train_data)/3000) + 1):
                    if i == 0:
                        save_out = sess.run(out, feed_dict={X: train_data[i*3000:(i+1)*3000], p_keep: 1.0})
                    else:
                        save_out_ = sess.run(out, feed_dict={X: train_data[i*3000:(i+1)*3000], p_keep: 1.0})
                        save_out = np.concatenate((save_out, save_out_))

                np.save('_job%s/save_out_teacher'%jobId, save_out) #[num_train_data, seq_len, 24]
                print('saved save_out_teacher.npy')


