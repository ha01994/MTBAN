#https://github.com/debbiemarkslab/EVcouplings/blob/develop/notebooks/running_jobs.ipynb
import os, sys, re, time
from Bio import SeqIO
import pandas as pd
from evcouplings.utils import read_config_file, write_config_file

jobId = sys.argv[1]
mut_file = sys.argv[2]
start_aa = int(sys.argv[3])
end_aa = int(sys.argv[4])
accession = sys.argv[5]



a = time.time() 
if os.path.exists("/home/ha01994/mutationTCN/_job%s/%s.a2m"%(jobId,jobId)) is False:
    
    fasta_file = "/home/ha01994/mutationTCN/_job%s/%s.fa"%(jobId,jobId)
    flag = False 
    trial = 0
    prev_cond = ''
    bit_score_threshold = 0.5

    while flag is False:
        
        trial += 1
        print('Trial %d..'%trial)
        print('bit_score_threshold: %f'%bit_score_threshold)

        output_dir = "/home/ha01994/mutationTCN/_job%s/%s"%(jobId,jobId)
        if os.path.exists(output_dir):
            os.system("rm -rf %s"%output_dir)

        config = read_config_file("/home/ha01994/mutationTCN/EVcouplings-develop/config/sample_config_monomer.txt", preserve_order=True)
        config["align"]["domain_threshold"] = bit_score_threshold
        config["align"]["sequence_threshold"] = bit_score_threshold
        config["global"]["prefix"] = output_dir
        config["global"]["sequence_id"] = jobId
        config["global"]["sequence_file"] = fasta_file
        config["global"]["region"] = [start_aa, end_aa]
        config["global"]["cpu"] = 2
        
        config_path = "/home/ha01994/mutationTCN/EVcouplings-develop/config/config_%s.txt"%jobId
        write_config_file(config_path, config)

        print('running evcouplings_runcfg...')
        os.system("/home/ha01994/anaconda3/bin/evcouplings_runcfg " + config_path)
        print('finished running evcouplings_runcfg...')

        data = pd.read_csv("/home/ha01994/mutationTCN/_job%s/%s/align/%s_alignment_statistics.csv"%(jobId, jobId, jobId))
        N_eff = data['N_eff'][0]
        num_seqs = data['num_seqs'][0]
        num_cov = data['num_cov'][0]
        perc_cov = data['perc_cov'][0]
        

        flag = True
        '''if N_eff >= num_cov: 
            logging.info('Condition satisfied')
            flag = True

        #elif N_eff < num_cov and perc_cov >= 0.8: 
        elif N_eff < num_cov: 
            logging.info('N_eff < num_cov')
            logging.info('N_eff: %f | num_cov: %d | perc_cov: %f'%(N_eff, num_cov, perc_cov))
            prev_cond = 'not_enough_seqs'
            bit_score_threshold -= 0.1

        elif N_eff >= num_cov and perc_cov < 0.8:
            logging.info('N_eff >= num_cov and perc_cov < 0.8')
            logging.info('N_eff: %f | num_cov: %d | perc_cov: %f'%(N_eff, num_cov, perc_cov))
            if prev_cond == 'not_enough_seqs':
                logging.info('Not enough sequences in the previous trial, but')
                logging.info('enough sequences in the current trial and perc_cov < 0.8')
                flag = True
            bit_score_threshold += 0.05

        else: #both problems
            logging.info('N_eff < num_cov and perc_cov < 0.8')
            logging.info('N_eff: %f | num_cov: %d | perc_cov: %f'%(N_eff, num_cov, perc_cov))
            prev_cond = ''
            bit_score_threshold -= 0.05'''

    b = time.time()

    print('DONE!')
    print('final bit_score_threshold: %f'%bit_score_threshold)
    print('N_eff: %f | num_cov: %d | perc_cov: %f'%(N_eff, num_cov, perc_cov))
    print('------------------------------------------------------------------')
    print('Time taken for building MSA (in hours): %f'%(round((b-a)/3600, 2)))

    
    os.system("cp /home/ha01994/mutationTCN/_job%s/%s/align/%s.a2m /home/ha01994/mutationTCN/_job%s/%s.a2m"%
              (jobId, jobId, jobId, jobId, jobId))
    os.system("cp /home/ha01994/mutationTCN/_job%s/%s/align/%s_alignment_statistics.csv "%(jobId, jobId, jobId)+
              "/home/ha01994/mutationTCN/_job%s/%s_alignment_statistics.csv"%(jobId, jobId))
    os.system("rm -rf /home/ha01994/mutationTCN/_job%s/%s/"%(jobId, jobId))
    os.system("rm %s"%config_path)
    os.system("rm /home/ha01994/mutationTCN/_job%s/%s.tar.gz"%(jobId, jobId))
    os.system("rm /home/ha01994/mutationTCN/_job%s/*.done"%jobId)
    os.system("rm /home/ha01994/mutationTCN/_job%s/*.outcfg"%jobId)
    
    os.system("cp /home/ha01994/mutationTCN/_job%s/%s.a2m /home/ha01994/databases/alignments/%s_%d-%d.a2m"%(jobId,jobId,accession,start_aa,end_aa))
    
