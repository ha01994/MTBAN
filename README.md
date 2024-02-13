# MTBAN
The current repository provides the source code and dataset for the paper: 

Kim HY, Jeon W, Kim D. An enhanced variant effect predictor based on a deep generative model and the Born-Again Networks. Scientific reports. 2021 Sep 27;11(1):1-7.

&nbsp;


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
This code is tested on python==3.6.5 and tensorflow-gpu==1.10.0. 

&nbsp;


## Available models
The source code offers two options:

**Option 1. mutationTCN**

This option predicts using the basic mutationTCN model, which is a deep generative model based on the Temporal Convolutional Network architecture.
This model takes a relatively short training time, yet gives results with reasonably high accuracy.

- [Reference] Kim HY, Kim D. Prediction of mutation effects using a deep temporal convolutional network. Bioinformatics. 2020 Apr 1;36(7):2047-52.

**Option 2. MTBAN**

This model is the optimization of the mutationTCN model with Born-Again Neural Networks (BAN).
The accuracy of the model is generally higher than the mutationTCN model.
However, this model takes a longer time to train.

- [Reference] Kim HY, Jeon W, Kim D. An enhanced variant effect predictor based on a deep generative model and the Born-Again Networks. Scientific reports. 2021 Sep 27;11(1):1-7.

&nbsp;


## Train model on sample dataset
Run ```python full_workflow.py``` to train the above models using a sample dataset (jobId 1; Uniprot accession P40692). The MSA file (```P40692_492-756.a2m```) and mutation file (```P40692_mutations.txt```) are provided in this folder. 

You can choose option 1 or option 2 when running ```python full_workflow.py```. 

The ```full_workflow.py``` file will sequentially run the following python files located in the ```src``` folder:

- ```build_msa.py```

- ```preprocess.py```

- ```train_option1.py``` (for option 1); ```train_teacher.py``` and ```train_student.py``` (for option 2)

If you want to test your own protein, you will need to either
1. prepare your own MSA (you need to make sure it covers the amino acid positions of the mutations of interest)
2. or build MSA using the ```src/build_msa.py``` file (for this you need to install EVcouplings - link: https://github.com/debbiemarkslab/EVcouplings). 

&nbsp;



## Description of the result file

In the result file (predictions_##.csv), the query protein UniProt accession is shown in the first line.

Also, in case where there are variants belonging to un-aligned columns in the alignment, those variants are indicated in the following lines. (After the alignment is generated, alignment columns that did not align - with more than 30% gaps - are dropped.)

Below that, the predictions are given in four columns: (1) score, (2) z-score, (3) probability of deleteriousness, and (4) predicted label.
The interpretation for each of them are as follows:

1. A score is computed as the log of the probability that the generative model assigns to a mutant sequence, divided by the probability assigned to the wild-type sequence (log(p_mutant/p_wt)). The smaller the score, the more likely the variant has damaging effect.

2. Z-scores are calculated by the z-score normalization of the distribution of scores for all possible missense variants against the target protein sequence.

3. The probability of deleteriousness refers to the probability of the variant being deleterious (ranges from 0 to 1). This is calculated by obtaining the z-scores for variants in the Humsavar database and computing the percentage of deleterious variants in different z-score ranges.

4. The 'predicted label' column is the label predicted by the model, either "deleterious" or "benign". The threshold was determined from the z-scores of the variants in the Humsavar database.

&nbsp;



## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.


&nbsp;
