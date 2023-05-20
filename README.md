# Simple_VPR_codebase

This repository serves as a starting point to implement a VPR pipeline. It allows you to train a simple
ResNet-18 on the GSV dataset. It relies on the [pytorch_metric_learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
library.

## Download datasets
NB: if you are using Colab, skip this section

The following script:

> python download_datasets.py

allows you to download GSV_xs, SF_xs, tokyo_xs, which are reduced version of the GSVCities, SF-XL, Tokyo247 datasets respectively.

   gsv_xs:stable angle for different time
   tokyo_xs:stable angle for different time 7/24
   sf_xs:diff angle for same time


## Install dependencies

You can install the required packages by running
> pip install -r requirements.txt


## Run an experiment
You can choose to validate/test on sf_xs or tokyo_xs.


>python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test

## Visually analize


##### Achitechture
Modal(basebone+final pooling)
 ->Loss function(+mining)
   ->Optimizer

##### REAL research
e.Modules:Gem-pooling layer
        NetVlad layer
        Mix-vpr

a.Miners: Pair margin
        Multi-Similarity
        GLOBAL PROXY-BASED HARD MINING

b.Losses:Contrasive
        Multi-Similarity

f.Optimizers:SGD
            ASGD
            ADAM
            ADAMW
