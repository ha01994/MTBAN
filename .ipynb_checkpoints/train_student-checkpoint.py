import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random, time, sys, os, csv
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
import tensorflow as tf
import numpy as np
from model import TemporalConvNet
import scipy
from scipy.special import softmax


jobId = sys.argv[1]
accession = sys.argv[2]
startpos = sys.argv[3]
endpos = sys.argv[4]



if True:

    data_aa = np.load('_job%s/data_aa.npy'%jobId) #(num_seqs, seq_len)
    wt_seq = np.reshape(data_aa[0,:], (1,-1))
    new_weights = np.load('_job%s/weights.npy'%jobId)    
    mutations = np.load('_job%s/mutations.npy'%jobId)
    all_possible_mut_data_aa = np.load('_job%s/all_possible_mut_data_aa.npy'%jobId)
    all_possible_mutations = np.load('_job%s/all_possible_mutations.npy'%jobId)

    temp = 4
    save_out = np.load('_job%s/save_out_teacher.npy'%jobId) #[num_train_data, seq_len, 24]
    save_out = save_out / temp #temperature
    save_out = softmax(save_out, axis=2)

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
        teacher_out = tf.placeholder(tf.float32, [None, seq_len, aa_size])

        out, wa_s = TemporalConvNet(aa_size, seq_len, [nhid]*n, k, p_keep)(X)
        #out: [num_train_data, seq_len, 24]

        label_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = X, logits = out), axis=1), (W/tf.reduce_mean(W))))

        teacher_loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels = teacher_out, logits = out/temp), axis=1), (W/tf.reduce_mean(W)))) #teacher loss

        loss = teacher_loss + label_loss

        test_out = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X, logits=out), axis=1)

        global_step = tf.Variable(0, trainable=False)

        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)    

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        print("Trainable parameters:")
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        print('starting training (student)...')
        sess.run(init)
        stored_exception=None

        for step in range(training_steps):
            X_b, W_b, t_b = next_batch_2(train_data, train_w, save_out, batch_size) 
            lr = 0.001
            sess.run(train_op, feed_dict={X: X_b, W: W_b, teacher_out: t_b, p_keep: keep_prob, learning_rate: lr})

            if step % 10000 == 0:
                print('step: %d'%step)            

            if step == training_steps - 1:
                wt_logp = -sess.run(test_out, feed_dict={X:wt_seq, p_keep: 1.0})
                mt_logp = np.array([])                        
                for i in range(0, int(len(all_possible_mut_data_aa)/3000) + 1):
                    mt_logp_ = -sess.run(test_out, feed_dict = {X: all_possible_mut_data_aa[i*3000:(i+1)*3000], 
                                         p_keep: 1.0})
                    mt_logp = np.concatenate((mt_logp, mt_logp_))

                all_possible_scores = mt_logp - wt_logp
                all_possible_scores = np.round(all_possible_scores, 4)
                
                normalized_values = z_score_normalize(all_possible_scores)
                probabilities = probability_prediction_option2(normalized_values)
                predictions = binary_prediction_option2(normalized_values)
                
                lz = list(zip(all_possible_mutations, all_possible_scores, normalized_values, 
                        probabilities, predictions))
                
                dd = accession +'_'+ str(startpos) +'-'+ str(endpos)

                
                mm = []; ss = []; zz = []; pp = []; pr = []
                for k in mutations:
                    for l in lz:
                        if l[0] == k:
                            mm.append(l[0])
                            ss.append(l[1])
                            zz.append(l[2])
                            pp.append(l[3])
                            pr.append(l[4])                
                            

                with open('_job%s/predictions_jobId%s.csv'%(jobId,jobId), 'a') as f:
                    f.write('Uniprot accession: %s\n'%accession)
                    f.write("variant,score,z-score,probability_of_deleteriousness,predicted_label\n")
                    for i in range(len(mm)):
                        f.write(str(mm[i])+','+str(ss[i])+','+str(zz[i])+','+str(pp[i])+','+str(pr[i])+'\n')
                