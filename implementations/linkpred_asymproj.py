"""Program for conducting Link Prediction over datasets downloaded from AsymProj

It constructs implicit matrix M^{WYS}, as described in our paper.

These datasets originally come from node2vec but we use the AsymProj train/test
partitions for consistency (node2vec, unfortunately, did not publish the splits
however they exactly decribed the partitioning procedure which was replicated
by AsymProj).

The program reads the dataset, trains the model (using functional SVD),
computes the AUC-ROC on test edges, and prints the AUC-ROC metric as well as
training time (i.e. spent inside the functional SVD computation).
"""

import json
import os
import sys
import time

from absl import app, flags
import numpy as np
import scipy.sparse
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tf_svd


if __name__ == '__main__':
  flags.DEFINE_string('dataset_name', 'ppi', 'Must be directory inside datasets_dir')
  flags.DEFINE_string(
      'datasets_dir', 'datasets/asymproj', 'Directory containing AsymProj datasets.')
  flags.DEFINE_integer('window', 5, 'Window for WYS approximation of DeepWalk.')
  flags.DEFINE_integer('dim', 30, '')
  flags.DEFINE_float('neg_coef', 3, '')
FLAGS = flags.FLAGS



def main(_):
  dataset_dir = os.path.join(os.path.expanduser(FLAGS.datasets_dir), FLAGS.dataset_name)

  train_edges = np.load(os.path.join(dataset_dir, 'train.txt.npy'))
  train_edges = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)
  spa = scipy.sparse.csr_matrix((np.ones([len(train_edges)]), (train_edges[:, 0], train_edges[:, 1]) ))
  mult_f = tf_svd.WYSDeepWalkPF(spa, window=FLAGS.window, mult_degrees=True,
                                neg_sample_coef=FLAGS.neg_coef)
  _ = tf_svd.fsvd(mult_f, 2, n_iter=1)  # Warm-up GPU
  print('training')
  started = time.time()
  u, s, v = tf_svd.fsvd(mult_f, FLAGS.dim, n_iter=20, n_redundancy=max(10, FLAGS.dim) )
  train_time = time.time() - started
  print('  ... done training')
  
  print('testing')
  U = (u*s).numpy()
  V = v.numpy()
  test_edges = np.load(os.path.join(dataset_dir, 'test.txt.npy'))
  test_neg_edges = np.load(os.path.join(dataset_dir, 'test.neg.txt.npy'))
  pos_scores = (U[test_edges[:, 0]] * V[test_edges[:, 1]]).sum(axis=1)
  neg_scores = (U[test_neg_edges[:, 0]] * V[test_neg_edges[:, 1]]).sum(axis=1)
  import sklearn.metrics
  all_score = np.concatenate([pos_scores, neg_scores], 0)
  all_y = np.concatenate([
      np.ones([len(pos_scores)], dtype='int32'),
      np.zeros([len(neg_scores)], dtype='int32'),
  ], 0)

  print(json.dumps({
    'auc': sklearn.metrics.roc_auc_score(all_y, all_score),
    'time': train_time,
  }))


if __name__ == '__main__':
  app.run(main)
