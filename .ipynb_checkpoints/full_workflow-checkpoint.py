import sys, os, csv, time, glob, re
#from utils import *
from Bio import SeqIO
import requests 
import random



##############################################################
jobId = '1'
accession = 'P40692'
option = 'option1'
#option = 'option2'
mut_file = "_job%s/%s_mutations.txt"%(jobId,jobId)
##############################################################


with open(mut_file) as f:
    mut_list = f.read().splitlines()

mutations = []
for m in mut_list:
    if len(m) >= 3:
        mut = m.strip()
        mutations.append(mut)
print('mutations', mutations)



os.system("wget https://www.uniprot.org/uniprot/%s.fasta"%accession)
os.system("mv ./%s.fasta _job%s/%s.fa"%(accession,jobId,jobId))
for rec in SeqIO.parse('_job%s/%s.fa'%(jobId,jobId), 'fasta'):
    sequence_length = len(rec)
    fasta_sequence = str(rec.seq)    


not_correspond = []
mut_positions = []
for i in mutations:
    split = re.split(r'(\d+)', str(i))
    if list(fasta_sequence)[int(split[1])-1] == split[0]: #The minus one is important
        mut_positions.append(int(split[1]))
    else:
        not_correspond.append(i)

if len(not_correspond) > 0:
    print('There are mutations which do not correctly correspond to the FASTA sequence of the protein')
    exit()
    



#########################################################################################################

# uncomment this if you need to build MSA (need to install EVcouplings)
'''
startpos = max(min(mut_positions) - 25, 1)
endpos = min(max(mut_positions) + 25, sequence_length)
if endpos - startpos < 20: 
    endpos = startpos + 20
elif endpos - startpos > 300: 
    endpos = startpos + 300
print('startpos', startpos)
print('endpos', endpos)

os.system("python src/build_msa.py %s %s %s %s %s"%(jobId, mut_file, startpos, endpos, accession))
'''
#########################################################################################################

# uncomment this if you need to preprocess
os.system("python src/preprocess.py %s"%(jobId))

#########################################################################################################
    
if option == "option1":
    print("starting train_option1.py")
    os.system("CUDA_VISIBLE_DEVICES=0 python src/train_option1.py %s %s"%(jobId, accession))

elif option == "option2":
    print("starting train_teacher.py")
    os.system("CUDA_VISIBLE_DEVICES=0 python src/train_teacher.py %s %s"%(jobId, accession))

    print("starting train_student.py")
    os.system("CUDA_VISIBLE_DEVICES=0 python src/train_student.py %s %s"%(jobId, accession))

#########################################################################################################




