# tf-fsvd
TensorFlow Implementation of Functional Singular Value Decomposition for paper
Fast Graph Learning

## Cite
If you find our code useful, you may cite us as:

    @inproceedings{haija2021fsvd,
      title={Fast Graph Learning with Unique Optimal Solutions},
      author={Sami Abu-El-Haija AND Valentino Crespi AND Greg Ver Steeg AND Aram Galstyan},
      year={2021},
      booktitle={arxiv},
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

 * File tf_svd.py contains the main logic for TensorFlow implementation of
   Functional SVD (function `fsvd`), as well as a few classes for constructing
   implicit matrices.
   * `SparseMatrixPF`: when implicit matrix is a pre-computed sparse matrix.
     Using this class, you can now enjoy the equivalent of `tf.linalg.svd` on
     sparse tensors.
   * `BlockWisePF`: when implicit matrix is is column-wise concatenation of other
     implicit matrices. The concatenation is computed by suppling a list of `ProductFn`'s
 * Directory implementations: contains implementations of simple methods employing `fsvd`.
 * Directory baselines: source code adapting competitive methods to produce metrics
   we report in our paper (time and accuracy).
 * Directory experiments: Shell scripts for running baselines and our implementations.
 * Directory results: Output directory containing results.


## Running Experiments

### Results
WRITE ME

### ROC-AUC Link Prediction over [AsymProj] datasets
WRITE ME

### Classification Experiments over [Planetoid] Citation datasets

WRITE ME

### HIT@20 over Drug-Drug Interaction Network
WRITE ME 
