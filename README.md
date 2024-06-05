
## Introduction

**Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation**


Federated recommendation aims to collect global knowledge by aggregating local models from massive devices, to provide recommendations while ensuring privacy. Current methods mainly leverage aggregation functions invented by federated vision community to aggregate parameters from similar clients, e.g., clustering aggregation. Despite considerable performance, we argue that it is suboptimal to apply them to federated recommendation directly. This is mainly reflected in the disparate model architectures. Different from structured parameters like convolutional neural networks in federated vision, federated recommender models usually distinguish itself by employing one-to-one item embedding table. Such a discrepancy induces the challenging embedding skew issue, which continually updates the trained embeddings but ignores the non-trained ones during aggregation, thus failing to predict future items accurately. To this end, we propose a personalized Federated recommendation model with Composite Aggregation (FedCA), which not only aggregates similar clients to enhance trained embeddings, but also aggregates complementary clients to update non-trained embeddings. Besides, we formulate the overall learning process into a unified optimization algorithm to jointly learn the similarity and complementarity. Extensive experiments on several real-world datasets substantiate the effectiveness of our proposed model.

## Requirements

The code is built on `Python=3.7` and `Pytorch=1.8`.

The other necessary Python libraries are as follows:
    
* coloredlogs>=15.0.1
* cvxpy==1.3.3
* numpy>=1.21.5
* pandas>=1.1.5
* scikit_learn>=1.0.2
* scipy>=1.7.3

To install these, please run the following commands:

  `pip install -r requirements.txt`
  
## Code Structure

The structure of our project is presented in a tree form as follows:

```
FedCA  # The root of project.
│   README.md
│   requirements.txt
│   train.py # The entry function file includes the main hyperparameter configurations.
|
└───datasets  # The used datasets in this work.
│   │   filmtrust   
|   │       ratings.dat
│   │   ml-100k   
|   │       ratings.dat
|   |   ...
|   |
└───model  # The main components in FR tasks.
│   │  engine.py # It includes the server aggregation and local training processes.
│   │  loss.py # Task-specific loss for local clients.
│   │  model.py # Defined backbone model (e.g., PMF and NCF) network architecture.
│   │  tools.py # Composite aggregation optimization process.
|   |
└───utils  # Other commonly used tools.
|   │   data.py # Codes related to data loading and preprocessing.
|   │   metrics.py # The evaluation metrics used in this work.
|   │   utils.py # Other utility functions.
```

## Parameters Settings

The meanings of the hyparameters are as follows:

`backbone`: the architecture of the backbone model used, the default value is `FCF`.

`dataset`: the name of used datasets, the default value is `filmtrust`.

`data_file `: the path of raw ratings data file, the default value is `ratings.dat`.

`train_frac`: the proportion of the training set used, the default value is `1.0`.

`clients_sample_ratio`: the proportion of user embeddings involved in the updates, the default value is `1.0`.

`global_round`: the number of global aggregation rounds, the default value is `100`.

`local_epoch`: the number of local training rounds, the default value is `10`.

`batch_size`: the number of local batch size, the default value is `256`.

`top_k`: the specific value of K in evaluation metrics, the default value is `10`.

`lr_structure`: the learning rate for training structured parameters, the default value is `1e-2`.

`lr_embedding`: the learning rate for training embedding parameters, the default value is `1e-2`.

`weight_decay`: the parameter regularization coefficient, the default value is `1e-3`.

`latent_dim`: the dimensions of user and item embeddings, the default value is `16`.

`mlp_layers`: the specific number of layers and units used in MLPs, the default value is `[32, 16, 8, 1]`.

`num_negative`: the number of negative samples used for local training, the default value is `4.0`.

`agg_clients_ratio`: the proportion used for participating in item embeddings aggregation, the default value is `0.6`.

`k_principal`: the number of singular value vectors used in SVD, the default value is `4.0`.

`alpha`: the weight of model similarity, the default value is `0.3`.

`beta`: the weight of data complementary, the default value is `0.3`.

`interpolation`: the specific coefficients of the interpolation method, the default value is `0.9`.


## Quick Start

Please change the used dataset and hyperparameters in `train.py`.

To run FCF with composite aggregation mode:

  `python train.py --backbone='FCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`

To run FedNCF with composite aggregation mode:

  `python train.py --backbone='FedNCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`
