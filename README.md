# MTBAN
The current repository provides the source code and dataset for the paper:

Kim HY, Jeon W, Kim D. An enhanced variant effect predictor based on a deep generative model and the Born-Again Networks. Scientific reports. 2021 Sep 27;11(1):1-7.

## Setting up environment
First create a conda environment, and activate the environment:
```
conda create -n mtban python=3.6.5
conda activate mtban
```
Clone this repository:
```
git clone https://github.com/ha01994/MTBAN.git
cd MTBAN
```
Then install the package requirements using pip:
```
pip install -r requirements.txt
```

## Available Models
The source code offers two options:

**Option 1. mutationTCN**

This option predicts using the basic mutationTCN model, which is a deep generative model based on the Temporal Convolutional Network architecture.
This model takes a relatively short training time, yet gives results with reasonably high accuracy.

Kim HY, Kim D. Prediction of mutation effects using a deep temporal convolutional network. Bioinformatics. 2020 Apr 1;36(7):2047-52.

**Option 2. MTBAN**

This model is the optimization of the mutationTCN model with Born-Again Neural Networks (BAN).
The accuracy of the model is generally higher than the mutationTCN model.
However, this model takes a longer time to train.



### 





## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.



