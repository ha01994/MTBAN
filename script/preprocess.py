#http://deeplearning.net/software/theano/tutorial/multi_cores.html
import numpy as np
from Bio import SeqIO
import time
import re
import theano
import theano.tensor as T
import os, sys, glob


jobId = sys.argv[1]



input_file = "_job%s/%s.a2m"%(jobId, jobId)
mut_file = "_job%s/%s_mutations.txt"%(jobId, jobId)

keys = list("ABCDEFGHIKLMNPQRSTVWXYZ-")
values = list(range(24))
dic = dict(zip(keys,values))
#print(dic)

fasta_seqs = SeqIO.parse(open(input_file),'fasta')
names,seqs=[],[]
for fasta in fasta_seqs: 
    name, seq = fasta.id, str(fasta.seq)
    names.append(name)
    seqs.append(seq)

print(names[0])    
print(seqs[0])

seq_range = names[0].split('/')[1]
start_pos = int(seq_range.split('-')[0])
end_pos = int(seq_range.split('-')[1])
print(start_pos)
print(end_pos)

pos_dic = list(zip(list(seqs[0]), list(range(start_pos, end_pos))))
#print('pos_dic', pos_dic)

pruned_seqs=[]
for seq in seqs:
    new_seq = ""
    for letter in list(seq):
        if letter!= "." and not (letter.islower()):
            new_seq += letter
    pruned_seqs.append(new_seq)

print('num of pruned sequences: %d'%len(pruned_seqs))
print('pruned sequence length: %d'%len(pruned_seqs[0]))


anchor_seq = pruned_seqs[0]
#print('anchor_seq: %s'%anchor_seq)


translated_seqs = []
for seq in pruned_seqs:
    new_seq = []
    for letter in list(seq):
        new_seq.append(dic[letter])
    translated_seqs.append(new_seq)
translated_seqs = np.array(translated_seqs)

np.save('_job%s/data_aa'%jobId, translated_seqs) 




pruned_positions = []
for pair in pos_dic:
    if pair[0]!= "." and not (pair[0].islower()):
        pruned_positions.append(pair[1])
#print('pruned_positions', pruned_positions)


#pruned_positions는 원래 sequence에서의 position
#list(range(len(pruned_seqs[0])))는 pruning 후의 position (consecutive)
new_dict = list(zip(list(anchor_seq), pruned_positions, list(range(len(pruned_seqs[0])))))
pos_pair_dict = dict(zip(pruned_positions, list(range(len(pruned_seqs[0])))))
#print('new_dict', new_dict)
#print('pos_pair_dict', pos_pair_dict)



##### all possible missense mutations wrt the target(anchor) sequence #####
mutations = []
mutated_seqs = []
for element in new_dict: #('K', 123, 0)
    vocabs = list("ACDEFGHIKLMNPQRSTVWY") #20 amino acids
    if element[0] in vocabs:
        vocabs.remove(element[0]) #19개
    else:
        pass
    for aa in vocabs:
        anchor = list(anchor_seq)
        mutations.append(element[0]+str(element[1])+aa)
        anchor[element[2]] = aa
        mut_seq = ''.join(anchor)
        mutated_seqs.append(mut_seq)
#print('number of all possible mutations', len(mutations))

mutations = np.array(mutations)

translated_mutated_seqs = []
for seq in mutated_seqs:
    new_seq = []
    for letter in list(seq):
        new_seq.append(dic[letter])
    translated_mutated_seqs.append(new_seq)

translated_mutated_seqs = np.array(translated_mutated_seqs)

np.save('_job%s/all_possible_mut_data_aa'%jobId, translated_mutated_seqs)
np.save('_job%s/all_possible_mutations'%jobId, mutations)




######### WEIGHT CALCULATION ###########
if os.path.exists('_job%s/weights.npy'%jobId) is False:
    print('calculating weights...')
    a = time.time()

    data_one_hot = np.eye(len(keys))[translated_seqs]
    x_train = data_one_hot
    theta = 0.2

    X = T.tensor3("x")
    cutoff = T.scalar("theta")
    X_flat = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    N_list, updates = theano.map(lambda x: 1.0 / T.sum(T.dot(X_flat, x) / T.dot(x, x) > 1 - cutoff), X_flat)
    weightfun = theano.function(inputs=[X, cutoff], outputs=[N_list], allow_input_downcast=True)
    weights = weightfun(x_train, theta)[0]

    print('finished calculating weights')
    b = time.time()
    print('time took in minutes: %f'%round((b-a)/60, 1))
    np.save('_job%s/weights'%jobId, weights)


    
    
######### process mutations ###########
with open(mut_file) as f:
    mut_list = f.read().splitlines()
    print('mut_list', mut_list)

mutations = []
for m in mut_list:
    if len(m) < 3:
        pass
    else:
        mut = m.strip()
        mutations.append(mut)
        

mut_seqs = []
new_mutations = []
for mutation in mutations: #G126A
    split = re.split(r'(\d+)', str(mutation))
    anchor = list(anchor_seq)
    if int(split[1]) in pruned_positions:
        anchor[pos_pair_dict[int(split[1])]] = split[2]
        seq = ''.join(anchor)
        mut_seqs.append(seq)
        new_mutations.append(mutation)
    else:
        pass


new_mutations = np.array(new_mutations)
np.save('_job%s/mutations'%jobId, new_mutations)




mut_file = "_job%s/%s_mutations.txt"%(jobId,jobId)

with open(mut_file) as f:
    mut_list = f.read().splitlines()

mutations = []
for m in mut_list:
    if len(m) >= 3:
        mut = m.strip()
        mutations.append(mut)
                  
mut_positions = []
for i in mutations:
    split = re.split(r'(\d+)', str(i))
    mut_positions.append(int(split[1]))

    

