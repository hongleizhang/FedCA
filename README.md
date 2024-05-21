
## Introduction

**Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation**

It is implemented by using Python and Pytorch.

The paper associated with these codes is currently under review by NeurIPS2024.

## Requirements

- The code is built on `python=3.7`

- all requirements are in the file `requirements.txt`. To install these, please run

  `pip install -r requirements.txt`

## Quick Start

- The folders should be firstly created in the root folder: `logs`.

- Please change the dataset folder in the line 26 of `train.py`.

- To run FCF with composite aggregation mode:

  `python train.py --alias='FCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`

- To run FedNCF with composite aggregation mode:

  `python train.py --alias='FedNCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`