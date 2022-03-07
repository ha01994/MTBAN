import sys, os, csv, time, glob, re
from utils import *
from Bio import SeqIO
import requests 

jobId = sys.argv[1]
mut_file = sys.argv[2]
accession = sys.argv[3]
option = sys.argv[4]

print('---------------------')
print('jobId',jobId)
print('mut_file',mut_file)
print('accession', accession)
print('option', option)
print('---------------------')


a = time.time()



with open('/home/ha01994/mutationTCN/_job%s/%s.txt'%(jobId,jobId)) as f:
    data = f.read().splitlines()

ziplist = list(zip(range(len(data)), data))
for (i,j) in ziplist:
    if j == "accession":
        accession = data[i+1].strip()
    elif j == "email":
        index_email = i
        email = data[i+1]
    
mutations_chunks = []
for (i,j) in ziplist:    
    if j == "mutations":
        index_mutations = i
        for l in range(i+1, index_email):
            mutations_chunks.append(data[l])
            
            

try:
    os.system("wget https://www.uniprot.org/uniprot/%s.fasta"%accession)
    os.system("mv ./%s.fasta /home/ha01994/mutationTCN/_job%s/%s.fa"%(accession,jobId,jobId))
    for rec in SeqIO.parse('/home/ha01994/mutationTCN/_job%s/%s.fa'%(jobId,jobId), 'fasta'):
        sequence_length = len(rec)
        fasta_sequence = str(rec.seq)
    print('sequence_length', sequence_length)
    #print(list(zip(list(range(sequence_length)), list(fasta_sequence))))    
    
except:
    print('The given accession is not a valid UniProt accession')
    with open('/home/ha01994/mutationTCN/_job%s/errorment.txt'%jobId,'w') as fw:
        fw.write('The given accession is not a valid UniProt accession')
        exit()
    
    


if mut_file == "null":
    with open("/home/ha01994/mutationTCN/_job%s/%s_mutations.txt"%(jobId,jobId), "w") as f:
        for chunk in mutations_chunks:
            f.write(chunk+'\n')
mut_file = "/home/ha01994/mutationTCN/_job%s/%s_mutations.txt"%(jobId,jobId)

with open(mut_file) as f:
    mut_list = f.read().splitlines()
    
mutations = []
for m in mut_list:
    if len(m) >= 3:
        mut = m.strip()
        mutations.append(mut)
print('mutations', mutations)

not_correspond=[]
mut_positions = []
for i in mutations:
    split = re.split(r'(\d+)', str(i))
    if list(fasta_sequence)[int(split[1])-1] == split[0]: #The minus one is important
        mut_positions.append(int(split[1]))
    else:
        not_correspond.append(i)

if len(not_correspond)>0 or '*' in ','.join(mutations):
    print('len(not_correspond)>0')
    with open('/home/ha01994/mutationTCN/_job%s/errorment.txt'%jobId,'w') as errorwrite:
        errorwrite.write('There are mutations which do not correctly correspond to the FASTA sequence of the protein with UniProt accession %s : %s'%(accession, ','.join(not_correspond)))
        exit()
    
print('mutations', mutations)
print('mut_positions', mut_positions)



###############################################################################################    
    
    

found_precomputed = False
if option == "option1":
    folders = glob.glob('/home/ha01994/databases/alignments_hp_100000it/*')
elif option == "option2":
    folders = glob.glob('/home/ha01994/databases/alignments_hp_updated_BAN/*')
    
for folder in folders:
    ds = folder.split('/')[-1] #Q9Y5E2_673-776
    aa = ds.split('_')[0]
    bb = int(ds.split('_')[1].split('-')[0])
    cc = int(ds.split('_')[1].split('-')[1])
    
    if aa == accession and bb <= min(mut_positions) and max(mut_positions) <= cc:
        print(folder)
        results = []
        for mut in mutations:
            with open(os.path.join(folder, 'predictions_%s.csv'%ds), 'r') as f:
                r = csv.reader(f)
                next(r)
                for line in r:
                    if line[0] == mut:
                        results.append(line)            
        if len(results) > 0:
            print('Found pre-computed results')
            print('results', results)
            found_precomputed = True
            
            with open('/home/ha01994/mutationTCN/_job%s/predictions_jobId%s.csv'%(jobId,jobId), 'w') as fw:
                fw.write('Uniprot accession: %s\n'%accession)
                found_mutations = [j[0] for j in results]
                excluded_mutations = [j for j in mutations if j not in found_mutations]
                if len(excluded_mutations) > 0:
                    fw.write('The following mutations correspond to positions that are not aligned in the multiple sequence alignment:\n') 
                    for i in excluded_mutations:
                        fw.write(i+'\n')
                    fw.write('\n\n')          
                    
                fw.write('variant,score,z-score,probability_of_deleteriousness,predicted_label\n')
                for r in results:
                    fw.write(','.join(r)+'\n')
                    
            break
        else: 
            found_precomputed = False                
        
        
        
        
import psutil
if found_precomputed == False:
    print('Did not find pre-computed results, starting the workflow')
    
    workflow_done=False
    while workflow_done == False:
        cpu_percent = psutil.cpu_percent()
        available_memory = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        
        if cpu_percent > 80 or available_memory < 20:
            print("waiting for enough CPU/memory...")
            time.sleep(3000)
            
        else:    
            startpos = max(min(mut_positions) - 25, 1)
            endpos = min(max(mut_positions) + 25, sequence_length)
            if endpos - startpos < 20: 
                endpos = startpos + 20
            elif endpos - startpos > 300: 
                endpos = startpos + 300
            print('startpos', startpos)
            print('endpos', endpos)  

            found_alignment = False
            alignments = glob.glob('/home/ha01994/databases/alignments/*')
            for alignment_path in alignments:
                alignment_name = alignment_path.split('/')[-1].split('.')[0]
                alignment_accession = alignment_name.split('_')[0]
                alignment_startpos = int(alignment_name.split('_')[1].split('-')[0])
                alignment_endpos = int(alignment_name.split('_')[1].split('-')[1])
                if accession == alignment_accession:
                    if alignment_startpos <= startpos and endpos <= alignment_endpos:
                        print("alignment already exists!")
                        os.system("cp %s /home/ha01994/mutationTCN/_job%s/%s.a2m"%(alignment_path, jobId, jobId))
                        found_alignment = True

            if found_alignment is False:
                os.system("/home/ha01994/anaconda3/bin/python3 /home/ha01994/mutationTCN/build_msa.py %s %s %s %s %s"%(jobId, mut_file, startpos, endpos, accession))



            os.system("/home/ha01994/anaconda3/bin/python3 /home/ha01994/mutationTCN/preprocess.py %s"%(jobId))    

            if os.path.exists('/home/ha01994/mutationTCN/_job%s/weights.npy'%jobId) is False:
                print("weights.npy not found")
                exit()

            if os.path.exists('/home/ha01994/mutationTCN/_job%s/errorment.txt'%jobId) is True:
                print('errorment.txt file exists')
                exit()



            training_done=False
            while training_done == False:
                memory = gpu_memory_map()
                print('memory', memory)
                if int(memory[0]) > 2000:
                    print("waiting for available GPU...")
                    time.sleep(2000)
                else:
                    if option == "option1":
                        print("starting train_option1.py")
                        os.system("/home/ha01994/anaconda3/bin/python3 /home/ha01994/mutationTCN/train_option1.py %s %s %s %s"%(jobId, accession, startpos, endpos))

                    elif option == "option2":
                        print("starting train_teacher.py")
                        os.system("CUDA_VISIBLE_DEVICES=0 /home/ha01994/anaconda3/bin/python3 /home/ha01994/mutationTCN/train_teacher.py %s %s %s %s"%(jobId, accession, startpos, endpos))
                        print("starting train_student.py")
                        os.system("CUDA_VISIBLE_DEVICES=0 /home/ha01994/anaconda3/bin/python3 /home/ha01994/mutationTCN/train_student.py %s %s %s %s"%(jobId, accession, startpos, endpos))
                    training_done = True
            
            workflow_done = True
            
            
b = time.time()
print('Total time taken (in hours): %f'%(round((b-a)/3600, 2)))
