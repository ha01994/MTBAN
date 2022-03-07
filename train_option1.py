import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random, time, sys, os, csv
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
import tensorflow as tf
import numpy as np
from model import TemporalConvNet


jobId = sys.argv[1]
accession = sys.argv[2]
startpos = sys.argv[3]
endpos = sys.argv[4]



if True:
    
    data_aa = np.load('/home/ha01994/mutationTCN/_job%s/data_aa.npy'%jobId) #(num_seqs, seq_len) 
    wt_seq = np.reshape(data_aa[0,:], (1,-1))
    weights = np.load('/home/ha01994/mutationTCN/_job%s/weights.npy'%jobId)    
    mutations = np.load('/home/ha01994/mutationTCN/_job%s/mutations.npy'%jobId)
    all_possible_mut_data_aa = np.load('/home/ha01994/mutationTCN/_job%s/all_possible_mut_data_aa.npy'%jobId)
    all_possible_mutations = np.load('/home/ha01994/mutationTCN/_job%s/all_possible_mutations.npy'%jobId)

    aa_size = 24
    seq_len = int(np.shape(data_aa)[1])
    k, n = return_k_n(seq_len)
    keep_prob = 0.7
    batch_size = 128
    starter_learning_rate = 0.001
    training_steps = 100000
    nhid = 128
    k, n = return_k_n(seq_len)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(10)
        X = tf.placeholder(tf.int32, [None, seq_len])
        W = tf.placeholder(tf.float32, [None])
        p_keep = tf.placeholder('float')

        out, wa_s = TemporalConvNet(aa_size, seq_len, [nhid]*n, k, p_keep)(X)

        loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                              labels=X, logits=out), axis=1), (W/tf.reduce_mean(W))))

        test_out = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X, logits=out), axis=1)

        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps=50000,
                                                   decay_rate=0.7, staircase=False)

        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)    

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        print("Trainable parameters:")
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))


    with tf.Session(graph=graph) as sess:
        print('starting training (option1)...')
        sess.run(init)
        stored_exception=None

        for step in range(training_steps):
            X_b, W_b = next_batch(data_aa, weights, batch_size) 
            sess.run(train_op, feed_dict={X: X_b, W: W_b, p_keep: keep_prob})

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
                probabilities = probability_prediction_option1(normalized_values)
                predictions = binary_prediction_option1(normalized_values)
                
                lz = list(zip(all_possible_mutations, all_possible_scores, normalized_values, 
                        probabilities, predictions))
                
                dd = accession +'_'+ str(startpos) +'-'+ str(endpos)
                os.system("mkdir /home/ha01994/databases/alignments_hp_100000it/%s"%dd)
                with open('/home/ha01994/databases/alignments_hp_100000it/%s/predictions_%s.csv'%(dd, dd), 'w') as f:
                    f.write("variant,score,z-score,probability_of_deleteriousness,prediction\n")
                    for line in lz:
                        f.write(str(line[0])+','+str(line[1])+','+str(line[2])+','+str(line[3])+','+str(line[4])+'\n')
                
                mm = []; ss = []; zz = []; pp = []; pr = []
                for k in mutations:
                    for l in lz:
                        if l[0] == k:
                            mm.append(l[0])
                            ss.append(l[1])
                            zz.append(l[2])
                            pp.append(l[3])
                            pr.append(l[4])                

                            
                with open('/home/ha01994/mutationTCN/_job%s/predictions_jobId%s.csv'%(jobId,jobId), 'a') as f:
                    f.write('Uniprot accession: %s\n'%accession)
                    f.write("variant,score,z-score,probability_of_deleteriousness,predicted_label\n")
                    for i in range(len(mm)):
                        f.write(str(mm[i])+','+str(ss[i])+','+str(zz[i])+','+str(pp[i])+','+str(pr[i])+'\n')
                
                with open('/home/ha01994/mutationTCN/_job%s/finished.txt'%jobId, 'w') as f:
                    f.write('Job is finished.')


