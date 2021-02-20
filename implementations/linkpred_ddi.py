
import os
import sys

from absl import app, flags
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import numpy as np
import scipy.sparse
import tensorflow as tf
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tf_fsvd

flags.DEFINE_string('dataset', 'ogbl-ddi', '')
flags.DEFINE_integer('hits', 20, 'DDI is evalued with hits@20')
flags.DEFINE_integer('wys_window', 2, 'Max power of transition matrix for WYS.')
flags.DEFINE_float('wys_neg_coef', 1, 'Negative co-efficient for WYS.')
flags.DEFINE_integer('svd_iters', 7, 'Number of svd iterations. Higher is better. 10 is usually close to perfect SVD.')

flags.DEFINE_integer('k', 100, 'Rank of SVD.')

flags.DEFINE_integer('num_runs', 10, 'number of training runs (to average accuracy)')
FLAGS = flags.FLAGS

def main(_):
  ds = LinkPropPredDataset(FLAGS.dataset)
  split_edge = ds.get_edge_split()
  train_edges = split_edge['train']['edge']
  train_edges = np.concatenate([train_edges, train_edges[:, ::-1]], axis=0)

  spa = scipy.sparse.csr_matrix((np.ones([len(train_edges)]), (train_edges[:, 0], train_edges[:, 1]) ))
  mult_f = tf_fsvd.WYSDeepWalkPF(
      spa, window=FLAGS.wys_window, mult_degrees=False, neg_sample_coef=FLAGS.wys_neg_coef)

  tt = tqdm.tqdm(range(FLAGS.num_runs))
  test_metrics = []
  val_metrics = []
  for run in tt:
    u, s, v = tf_fsvd.fsvd(mult_f, FLAGS.k, n_iter=FLAGS.svd_iters, n_redundancy=FLAGS.k*3)
    
    dataset = LinkPropPredDataset(FLAGS.dataset)
    evaluator = Evaluator(name=FLAGS.dataset)
    evaluator.K = FLAGS.hits 
    split_edge = dataset.get_edge_split()
   
    metrics = []
    for split in ('test', 'valid'):
      pos_edges = split_edge[split]['edge']
      neg_edges = split_edge[split]['edge_neg']
      
      pos_scores = tf.reduce_sum(
          tf.gather(u * s, pos_edges[:, 0]) * tf.gather(v, pos_edges[:, 1]),
          axis=1).numpy()
      neg_scores = tf.reduce_sum(
          tf.gather(u * s, neg_edges[:, 0]) * tf.gather(v, neg_edges[:, 1]),
          axis=1).numpy()
      metric = evaluator.eval({'y_pred_pos': pos_scores, 'y_pred_neg': neg_scores})
      metrics.append(metric['hits@%i' % FLAGS.hits])
    test_metrics.append(metrics[0])
    val_metrics.append(metrics[1])
    
    tt.set_description('HITS@%i: validate=%g; test=%g' % (
      FLAGS.hits, np.mean(val_metrics), np.mean(test_metrics)))

  print('\n\n *** Trained for %i times and average metrics are:')
  print('HITS@20 test: mean=%g; std=%g' % (np.mean(test_metrics), np.std(test_metrics)))
  print('HITS@20 validate: mean=%g; std=%g' % (np.mean(val_metrics), np.std(val_metrics)))


if __name__ == '__main__':
  app.run(main)
