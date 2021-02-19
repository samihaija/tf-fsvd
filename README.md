# tf-fsvd
TensorFlow Implementation of Functional Singular Value Decomposition for paper
[**Fast Graph Learning with Unique Optimal Solutions**](https://arxiv.org/abs/2102.08530)

## Cite
If you find our code useful, you may cite us as:

    @inproceedings{haija2021fsvd,
      title={Fast Graph Learning with Unique Optimal Solutions},
      author={Sami Abu-El-Haija AND Valentino Crespi AND Greg Ver Steeg AND Aram Galstyan},
      year={2021},
      booktitle={arxiv:2102.08530},
    }

## Introduction

This codebase contains TensorFlow implementation of Functional SVD, an SVD routine
that accepts objects with 3 attributes: `dot`, `T`, and `shape`.
The object must be able to exactly multiply an (implicit) matrix `M` by any other
matrix. Specifically, it should implement:

  1. `dot(M1)`: should return `M @ M1`
  1. `T`: property should return another object that (implicitly) contains transpose of `M`.
  1. `shape`: property should return the shape of the (implicit) matrix `M`.

In most practical cases, `M` is implicit i.e. need not to be exactly computed.
For consistency, such objects could inherit the abstract class `ProductFn`.



## Simple Usage Example

Suppose you have an explicit sparse matrix `mat`

    import scipy.sparse
    import tf_svd

    m = scipy.sparse.csr_mat( ... )
    fn = tf_svd.SparseMatrixPF(m)

    u, s, v = tf_svd.fsvd(fn, k=20)  # Rank 20 decomposition


The intent of this utility is for implicit matrices. For which, you may implement
your own `ProductFn` class. You can take a look at `BlockWisePF` or `WYSDeepWalkPF`.


## File Structure / Documentation

 * File [tf_svd.py](https://github.com/samihaija/tf-fsvd/blob/main/tf_svd.py) contains the main logic for TensorFlow implementation of
   Functional SVD (function `fsvd`), as well as a few classes for constructing
   implicit matrices.
   * `SparseMatrixPF`: when implicit matrix is a pre-computed sparse matrix.
     Using this class, you can now enjoy the equivalent of `tf.linalg.svd` on
     sparse tensors.
   * `BlockWisePF`: when implicit matrix is is column-wise concatenation of other
     implicit matrices. The concatenation is computed by suppling a list of `ProductFn`'s
 * Directory [implementations](https://github.com/samihaija/tf-fsvd/tree/main/implementations): contains implementations of simple methods employing `fsvd`.
 * Directory [baselines](https://github.com/samihaija/tf-fsvd/tree/main/baselines): source code adapting competitive methods to produce metrics
   we report in our paper (time and accuracy).
 * Directory [experiments](https://github.com/samihaija/tf-fsvd/tree/main/experiments): Shell scripts for running baselines and our implementations.
 * Directory [results](https://github.com/samihaija/tf-fsvd/tree/main/results): Output directory containing results.


## Running Experiments

### ROC-AUC Link Prediction over AsymProj/WYS datasets
The AsymProj datasets are located in directory [datasets/asymproj](https://github.com/samihaija/tf-fsvd/tree/main/datasets/asymproj).

You can run the script for training on AsympProj datasets and measuring test ROC-AUC as:

    python3 implementations/linkpred_asymproj.py

You can append flag `--help` to above to see which flags you can set for changing the dataset or the SVD rank.

You can run sweep on svd rank, for each of those datasets, by invoking:

    # Sweep fSVD rank (k) on 4 link pred datasets. Make 3 runs per (dataset, k)
    # Time is dominated by statement `import tensorflow as tf`
    python3 experiments/fsvd_linkpred_k_sweep.py | bash  # You may remove "| bash" if you want to hand-pick commands.
    
    # Summarize results onto CSV
    python3 experiments/summarize_svdf_linkpred_sweep.py > results/linkpred_d_sweep/fsvd.csv
    
    # Plot the sweep curve
    python3 experiments/plot_sweep_k_linkpred.py


and running all printed commands. Alternatively, you can pipe the output of above to bash. This should populate directory `results/linkpred_d_sweep/fsvd/`.

#### Baselines

 * You can run the Watch Your Step baseline as:
   
        bash experiments/baselines/run_wys.sh
   
   which runs only once for every link prediction dataset. Watch Your Step spends some time computing the transition matrix    powers (T^2, .., T^5).

 * You can run NetMF baselines (both approximate and exact) as:
 
       bash experiments/baselines/run_netmf.sh
   
 * You can run node2vec baseline as:
 
       experiments/baselines/run_n2v.sh


### Classification Experiments over Planetoid Citation datasets
These datasets are from the planetoid paper. To obtain them, you should clone their repo:

    mkdir -p ~/data
    cd ~/data
    git clone git@github.com:kimiyoung/planetoid.git

You can run the script for training and testing on planetoid datasets as:

    python3 implementations/node_ssc_planetoid.py

You can append flag `--help` to above to see which flags you can set for changing the dataset or the number of layers.

You can sweep the number of layers and other hyperparameters by running this script:

    python3 experiments/planetoid_hp_search.py # Directly invokes python many times

If you only want to see the commands being run by above script, but **not** run the commands, you can:


### HIT@20 over Drug-Drug Interaction Network
WRITE ME 

