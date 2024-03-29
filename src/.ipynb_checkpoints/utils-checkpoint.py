import os, csv
import numpy as np


def next_batch(x, y, batch_size):
    index = np.arange(len(x))
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index] #(batch_size, seq_len)


def next_batch_2(x, y, z, batch_size):
    index = np.arange(len(x))
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index], z[random_index] #(batch_size, seq_len)


def return_k_n(seq_len):
    with open('/home/ha01994/MTBAN/src/k_n.csv', 'r') as f:
        r = csv.reader(f)
        next(r)
        for line in r:
            if int(line[2]) >= seq_len:
                k = int(line[0])
                n = int(line[1])
                break
            else:
                continue
    return k,n


def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(np.round(normalized_num, 4))
    return normalized


####################################################################################################


prob_dist_option1 = [[[-36.9612, -36.4458], 1.0], [[-36.4458, -35.9304], 1.0], [[-35.9304, -35.415], 1.0], [[-35.415, -34.8996], 1.0], [[-34.8996, -34.3842], 1.0], [[-34.3842, -33.8688], 1.0], [[-33.8688, -33.3534], 1.0], [[-33.3534, -32.838], 1.0], [[-32.838, -32.3226], 1.0], [[-32.3226, -31.8072], 1.0], [[-31.8072, -31.2918], 1.0], [[-31.2918, -30.7764], 1.0], [[-30.7764, -30.261], 1.0], [[-30.261, -29.7456], 1.0], [[-29.7456, -29.2302], 1.0], [[-29.2302, -28.7148], 1.0], [[-28.7148, -28.1994], 1.0], [[-28.1994, -27.684], 1.0], [[-27.684, -27.1686], 1.0], [[-27.1686, -26.6532], 1.0], [[-26.6532, -26.1378], 1.0], [[-26.1378, -25.6224], 1.0], [[-25.6224, -25.107], 1.0], [[-25.107, -24.5916], 1.0], [[-24.5916, -24.0762], 1.0], [[-24.0762, -23.5608], 1.0], [[-23.5608, -23.0454], 1.0], [[-23.0454, -22.53], 1.0], [[-22.53, -22.0146], 1.0], [[-22.0146, -21.4992], 1.0], [[-21.4992, -20.9838], 1.0], [[-20.9838, -20.4684], 1.0], [[-20.4684, -19.953], 1.0], [[-19.953, -19.4376], 1.0], [[-19.4376, -18.9222], 1.0], [[-18.9222, -18.4068], 1.0], [[-18.4068, -17.8914], 1.0], [[-17.8914, -17.376], 1.0], [[-17.376, -16.8606], 1.0], [[-16.8606, -16.3452], 1.0], [[-16.3452, -15.8298], 1.0], [[-15.8298, -15.3144], 1.0], [[-15.3144, -14.799], 1.0], [[-14.799, -14.2836], 1.0], [[-14.2836, -13.7682], 1.0], [[-13.7682, -13.2528], 1.0], [[-13.2528, -12.7374], 1.0], [[-12.7374, -12.222], 1.0], [[-12.222, -11.7066], 1.0], [[-11.7066, -11.1912], 1.0], [[-11.1912, -10.6758], 1.0], [[-10.6758, -10.1604], 1.0], [[-10.1604, -9.645], 1.0], [[-9.645, -9.1296], 1.0], [[-9.1296, -8.6142], 1.0], [[-8.6142, -8.0988], 1.0], [[-8.0988, -7.5834], 1.0], [[-7.5834, -7.068], 1.0], [[-7.068, -6.5526], 1.0], [[-6.5526, -6.0372], 1.0], [[-6.0372, -5.5218], 0.8749], [[-5.5218, -5.0064], 0.9998], [[-5.0064, -4.491], 0.8999], [[-4.491, -3.9756], 0.8666], [[-3.9756, -3.4602], 0.8571], [[-3.4602, -2.9448], 0.8043], [[-2.9448, -2.4294], 0.8829], [[-2.4294, -1.914], 0.8635], [[-1.914, -1.3986], 0.8737], [[-1.3986, -0.8832], 0.864], [[-0.8832, -0.3678], 0.8272], [[-0.3678, 0.1476], 0.7523], [[0.1476, 0.663], 0.5035], [[0.663, 1.1784], 0.2461], [[1.1784, 1.6938], 0.0955], [[1.6938, 2.2092], 0.0183], [[2.2092, 2.7246], 0.0038], [[2.7246, 3.24], 0.0], [[3.24, 3.7554], 0.0], [[3.7554, 4.2708], 0.0], [[4.2708, 4.7862], 0.0], [[4.7862, 5.3016], 0.0], [[5.3016, 5.817], 0.0], [[5.817, 6.3324], 0.0], [[6.3324, 6.8478], 0.0], [[6.8478, 7.3632], 0.0], [[7.3632, 7.8786], 0.0], [[7.8786, 8.394], 0.0], [[8.394, 8.9094], 0.0], [[8.9094, 9.4239], 0.0]]

def probability_prediction_option1(lst):
    prob_preds = []
    for z in lst:
        if z < prob_dist_option1[0][0][0]:
            prob_preds.append(1.0)
        elif z > prob_dist_option1[-1][0][1]:
            prob_preds.append(0.0)
        else:
            for p in prob_dist_option1:
                if p[0][0] <= z < p[0][1]:
                    prob_preds.append(p[1])
        
    return prob_preds


def binary_prediction_option1(lst):
    binary_preds = []
    for z in lst:
        if z >= 0.663:
            binary_preds.append('benign')
        else:
            binary_preds.append('deleterious')
    return binary_preds
    

####################################################################################################

prob_dist_option2 = [[[-12.2228, -12.0465], 1.0], [[-12.0465, -11.8702], 1.0], [[-11.8702, -11.6939], 1.0], [[-11.6939, -11.5176], 1.0], [[-11.5176, -11.3413], 1.0], [[-11.3413, -11.165], 1.0], [[-11.165, -10.9887], 1.0], [[-10.9887, -10.8124], 1.0], [[-10.8124, -10.6361], 1.0], [[-10.6361, -10.4598], 1.0], [[-10.4598, -10.2835], 1.0], [[-10.2835, -10.1072], 1.0], [[-10.1072, -9.9309], 1.0], [[-9.9309, -9.7546], 1.0], [[-9.7546, -9.5783], 1.0], [[-9.5783, -9.402], 1.0], [[-9.402, -9.2257], 1.0], [[-9.2257, -9.0494], 1.0], [[-9.0494, -8.8731], 1.0], [[-8.8731, -8.6968], 1.0], [[-8.6968, -8.5205], 1.0], [[-8.5205, -8.3442], 1.0], [[-8.3442, -8.1679], 1.0], [[-8.1679, -7.9916], 1.0], [[-7.9916, -7.8153], 1.0], [[-7.8153, -7.639], 1.0], [[-7.639, -7.4627], 1.0], [[-7.4627, -7.2864], 1.0], [[-7.2864, -7.1101], 1.0], [[-7.1101, -6.9338], 1.0], [[-6.9338, -6.7575], 1.0], [[-6.7575, -6.5812], 1.0], [[-6.5812, -6.4049], 1.0], [[-6.4049, -6.2286], 1.0], [[-6.2286, -6.0523], 1.0], [[-6.0523, -5.876], 0.0], [[-5.876, -5.6997], 0.0], [[-5.6997, -5.5234], 0.0], [[-5.5234, -5.3471], 0.0], [[-5.3471, -5.1708], 0.0], [[-5.1708, -4.9945], 0.999], [[-4.9945, -4.8182], 0.0], [[-4.8182, -4.6419], 0.999], [[-4.6419, -4.4656], 0.0], [[-4.4656, -4.2893], 0.0], [[-4.2893, -4.113], 0.999], [[-4.113, -3.9367], 0.0], [[-3.9367, -3.7604], 0.999], [[-3.7604, -3.5841], 0.999], [[-3.5841, -3.4078], 0.999], [[-3.4078, -3.2315], 0.9995], [[-3.2315, -3.0552], 0.999], [[-3.0552, -2.8789], 0.999], [[-2.8789, -2.7026], 0.9995], [[-2.7026, -2.5263], 0.9998], [[-2.5263, -2.35], 0.8888], [[-2.35, -2.1737], 0.857], [[-2.1737, -1.9974], 0.909], [[-1.9974, -1.8211], 0.7999], [[-1.8211, -1.6448], 0.875], [[-1.6448, -1.4685], 0.8333], [[-1.4685, -1.2922], 0.9302], [[-1.2922, -1.1159], 0.8444], [[-1.1159, -0.9396], 0.913], [[-0.9396, -0.7633], 0.9365], [[-0.7633, -0.587], 0.8108], [[-0.587, -0.4107], 0.7975], [[-0.4107, -0.2344], 0.8257], [[-0.2344, -0.0581], 0.7581], [[-0.0581, 0.1182], 0.7764], [[0.1182, 0.2945], 0.6603], [[0.2945, 0.4708], 0.5], [[0.4708, 0.6471], 0.5], [[0.6471, 0.8234], 0.2718], [[0.8234, 0.9997], 0.2167], [[0.9997, 1.176], 0.1718], [[1.176, 1.3523], 0.1229], [[1.3523, 1.5286], 0.061], [[1.5286, 1.7049], 0.0385], [[1.7049, 1.8812], 0.0], [[1.8812, 2.0575], 0.0323], [[2.0575, 2.2338], 0.0], [[2.2338, 2.4101], 0.0], [[2.4101, 2.5864], 0.0], [[2.5864, 2.7627], 0.0], [[2.7627, 2.939], 0.0], [[2.939, 3.1153], 0.0], [[3.1153, 3.2916], 0.0], [[3.2916, 3.4679], 0.0], [[3.4679, 3.6436], 0.0]]

def probability_prediction_option2(lst):
    prob_preds = []
    for z in lst:
        if z < prob_dist_option2[0][0][0]:
            prob_preds.append(1.0)
        elif z > prob_dist_option2[-1][0][1]:
            prob_preds.append(0.0)
        else:
            for p in prob_dist_option2:
                if p[0][0] <= z < p[0][1]:
                    prob_preds.append(p[1])
        
    return prob_preds


def binary_prediction_option2(lst):
    binary_preds = []
    for z in lst:
        if z >= 0.647:
            binary_preds.append('benign')
        else:
            binary_preds.append('deleterious')
    return binary_preds

####################################################################################################
    
    
import subprocess, re

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu
