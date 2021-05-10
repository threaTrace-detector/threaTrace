# THREATRACE

## Overview

This repository contains the evaluation reproduction material and guideline for the **THREATRACE**'s paper. The complete detection system will be released to the community soon. 

## Evaluation reproduction

This is the guideline for reproducing the evaluation in the paper. **THREATRACE** has the implementation of the Ubuntu system currently.

### Clone the repository and pre-requirement

  1. Clone the complete repository into a directory (named as ROOT, the same below) of your Ubuntu system.

  2. Deploy PyG, a geometric deep learning extension library for PyTorch. The installation guideline can be found here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

  3. Other required Python packages: 
  
    numpy, pandas, argparse, subprocess, os, sys, time, psutil, random, csv, re


### Datasets preparation

Due to space constraints, we cannot store the parsed datasets in this repository. Therefore, the first step of the evaluation reproduction is to prepare the datasets. 

> StreamSpot dataset

   Download the StreamSport dataset from: https://github.com/sbustreamspot/sbustreamspot-data

    You need to download: 
        all.tar.gz

   Copy the dataset to ROOT/threaTrace/graphchi-cpp-master/graph_data/streamspot/: 
  
    cp all.tar.gz ROOT/threaTrace/graphchi-cpp-master/graph_data/streamspot/

   Run the parsing script:
 
    cd ROOT/threaTrace/scripts
    python parse_streamspot.py


> Unicorn SC-2 dataset

   Download the Unicorn SC-2 dataset from: https://github.com/margoseltzer/shellshock-apt

    You need to download: 
        camflow-benign-*
        camflow-attack-*

   Copy the dataset to ROOT/threaTrace/graphchi-cpp-master/graph_data/unicornsc/: 
  
    cp camflow-benign-* ROOT/threaTrace/graphchi-cpp-master/graph_data/unicornsc/
    cp camflow-attack-* ROOT/threaTrace/graphchi-cpp-master/graph_data/unicornsc/

   Run the parsing script:
 
    cd ROOT/threaTrace/scripts
    python parse_unicornsc.py
    
> DARPA TC dataset
    
   Download the DARPA TC dataset from: https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-
 
      You need to download:
          cadets/ta1-cadets-e3-official.json.tar.gz
          cadets/ta1-cadets-e3-official-2.json.tar.gz
          fivedirections/ta1-fivedirections-e3-official-2.json.tar.gz
          theia/ta1-theia-e3-official-1r.json.tar.gz
          theia/ta1-theia-e3-official-6r.json.tar.gz
          trace/ta1-trace-e3-official-1.json.tar.gz
      
   Copy the dataset to ROOT/threaTrace/graphchi-cpp-master/graph_data/darpatc/: 
  
    cp ta1-* ROOT/threaTrace/graphchi-cpp-master/graph_data/darpatc/

   Run the parsing script:
 
    cd ROOT/threaTrace/scripts
    python parse_darpatc.py


### Evaluation in StreamSpot dataset

We provide example models in `ROOT/threaTrace/example_models/streamspot` for convenience. You can also choose to train the models. 

> Use example models

    cd ROOT/threaTrace/scripts
    python setup.py
    cp ROOT/threaTrace/example_models/streamspot/* ROOT/threaTrace/models/
    cp ROOT/threaTrace/example_models/streamspot/models_list.txt ./
    cp ROOT/threaTrace/example_models/streamspot/run_* ./
  
> Or train models

    cd ROOT/threaTrace/scripts
    python setup.py
    python train_streamspot.py
    
The training procedure may take some time.

> Test

Once the models are prepared, you can start testing **THREATRACE** in StreamSpot dataset.


    cd ROOT/threaTrace/scripts
    chmod 777 run_benign.sh
    chmod 777 run_attack.sh
    ./run_benign.sh
    ./run_attack.sh

When the testing procedure finishes, use the following command to evaluate the detection performance in StreamSpot dataset.

    python evaluate_streamspot.py

### Evaluation in Unicorn SC-2 dataset

We provide example models in `ROOT/threaTrace/example_models/unicornsc` for convenience. You can also choose to train the models. 

> Use example models

    cd ROOT/threaTrace/scripts
    python setup.py
    cp ROOT/threaTrace/example_models/unicornsc/* ROOT/threaTrace/models/
    cp ROOT/threaTrace/example_models/unicornsc/models_list.txt ./
    cp ROOT/threaTrace/example_models/unicornsc/run_* ./

> Or train models

    cd ROOT/threaTrace/scripts
    python setup.py
    python train_unicornsc.py
    
The training procedure may take some time.

> Test

Once the models are prepared, you can start testing **THREATRACE** in Unicorn SC-2 dataset.

    cd ROOT/threaTrace/scripts
    chmod 777 run_benign.sh
    chmod 777 run_attack.sh
    ./run_benign.sh
    ./run_attack.sh

When the testing procedure finishes, use the following command to evaluate the detection performance in the Unicorn SC-2 dataset.

    python evaluate_unicornsc.py

### Evaluation in DARPA TC dataset

We provide example models in `ROOT/threaTrace/example_models/darpatc` for convenience. You can also choose to train the models. 

> Use example models. Choose the scene X: cadets/trace/theia/fivedirections

    cd ROOT/threaTrace/scripts
    python setup.py
    cp ROOT/threaTrace/example_models/darpatc/X/* ROOT/threaTrace/models/

> Or train models. Choose the scene X: cadets/trace/theia/fivedirections

    cd ROOT/threaTrace/scripts
    python setup.py
    python train_darpatc.py --scene X
    
The training procedure may take some time.

> Test

Once the models are prepared, you can start testing **THREATRACE** in DARPA TC dataset. Set X as the same as the value in the training phase.

    python test_darpatc.py --scene X

When the testing procedure finishes, use the following command to evaluate the detection performance in DARPA TC dataset.

    python evaluate_darpatc.py


## License

This project is licensed under the MIT License - see the LICENSE file for details
