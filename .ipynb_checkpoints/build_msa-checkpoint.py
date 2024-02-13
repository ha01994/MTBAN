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
if True:
    
    fasta_file = "_job%s/%s.fa"%(jobId,jobId)
    flag = False 
    trial = 0
    prev_cond = ''
    bit_score_threshold = 0.5

    while flag is False:
        
        trial += 1
        print('Trial %d..'%trial)
        print('bit_score_threshold: %f'%bit_score_threshold)

        output_dir = "_job%s/%s"%(jobId,jobId)
        if os.path.exists(output_dir):
            os.system("rm -rf %s"%output_dir)

        config = read_config_file("EVcouplings-develop/config/sample_config_monomer.txt", preserve_order=True)
        config["align"]["domain_threshold"] = bit_score_threshold
        config["align"]["sequence_threshold"] = bit_score_threshold
        config["global"]["prefix"] = output_dir
        config["global"]["sequence_id"] = jobId
        config["global"]["sequence_file"] = fasta_file
        config["global"]["region"] = [start_aa, end_aa]
        config["global"]["cpu"] = 2
        
        config_path = "EVcouplings-develop/config/config_%s.txt"%jobId
        write_config_file(config_path, config)

        print('running evcouplings_runcfg...')
        os.system("/home/ha01994/miniconda3/bin/evcouplings_runcfg " + config_path)
        print('finished running evcouplings_runcfg...')

        data = pd.read_csv("_job%s/%s/align/%s_alignment_statistics.csv"%(jobId, jobId, jobId))
        N_eff = data['N_eff'][0]
        num_seqs = data['num_seqs'][0]
        num_cov = data['num_cov'][0]
        perc_cov = data['perc_cov'][0]
        

        flag = True

    b = time.time()

    print('DONE!')
    print('final bit_score_threshold: %f'%bit_score_threshold)
    print('N_eff: %f | num_cov: %d | perc_cov: %f'%(N_eff, num_cov, perc_cov))
    print('------------------------------------------------------------------')
    print('Time taken for building MSA (in hours): %f'%(round((b-a)/3600, 2)))

    
    os.system("cp _job%s/%s/align/%s.a2m _job%s/%s.a2m"%(jobId, jobId, jobId, jobId, jobId))
    os.system("cp _job%s/%s/align/%s_alignment_statistics.csv "%(jobId, jobId, jobId)+"_job%s/%s_alignment_statistics.csv"%(jobId, jobId))
    os.system("rm -rf _job%s/%s/"%(jobId, jobId))
    os.system("rm %s"%config_path)
    os.system("rm _job%s/%s.tar.gz"%(jobId, jobId))
    os.system("rm _job%s/*.done"%jobId)
    os.system("rm _job%s/*.outcfg"%jobId)
    
    
    
