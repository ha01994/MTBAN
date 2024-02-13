# MTBAN
The current repository provides the source code and dataset for the paper:

Kim HY, Jeon W, Kim D. An enhanced variant effect predictor based on a deep generative model and the Born-Again Networks. Scientific reports. 2021 Sep 27;11(1):1-7.

## Setting up environment for running MTBAN
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


## Running code on an example dataset
Example training, validation, and test data can be found in the `example_run/data` folder. 

Example format of the files in `example_run/data`:

    peptide,A1,A2,A3,B1,B2,B3,binder
    NLVPMVATV,SVFSS,VVTGGEV,AGPEGGYSGAGSYQLT,SGDLS,YYNGEE,ASSVSGATADTQY,0
    LLWNGPMAV,TRDTTYY,RNSFDEQN,ALSGEGTGRRALT,GTSNPN,SVGIG,AWSVQGTDTQY,0
    TTDPSFLGRY,TSGFNG,NVLDGL,AVRVFNARLM,SNHLY,FYNNEI,ASSEEIAKNIQY,1
    GILGFVFTL,VSGLRG,LYSAGEE,AVRANQAGTALI,SGHRS,YFSETQ,ASSLTGSNTEAF,1

### Data preprocessing
To change the data format to fit our data processing pipeline, run:
```
python get_data_ready.py
```
Example format of the formatted data:

    pep_id,tcr_id,label,split
    pep12,tcr254,0,train    
    pep16,tcr3719,0,val
    pep3,tcr2713,1,train
    pep4,tcr295,1,val
    
The sequences corresponding to the peptide and TCR IDs are found in `example_run/formatted_data/ids_pep.csv` and `example_run/formatted_data/ids_tcr.csv`.

To perform preprocessing on the formatted data, run:
```
python pp.py
```
This will generate preprocessed files (in .npy and .pkl format) in the `example_run/features` folder. These files are used for training and testing.





## Contact information
If you have questions, please contact ha01994@kaist.ac.kr.



