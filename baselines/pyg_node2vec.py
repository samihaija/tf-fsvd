
import os
import os.path as osp
import time

from absl import flags, app 

import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

import numpy as np


flags.DEFINE_string(
    'dataset', 'ppi', 'Dataset name. One of AsymProj datasets.')
flags.DEFINE_string('datasets_dir', 'datasets/asymproj',
                    'Directory where all AsymProj datasets live, including --dataset')
flags.DEFINE_integer(
    'dim', 64, 'Dim of embedding')
flags.DEFINE_integer('C', 5, 'context window.')
flags.DEFINE_string('output_dir', 'results/baselines/pyg-node2vec', 'directory to output results')
flags.DEFINE_string('run', '', 'If set, will be added to output filename.')

FLAGS = flags.FLAGS

def main(_):
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  start = time.time()
  outfile = os.path.join(FLAGS.output_dir, '%s_%i_%i' % (FLAGS.dataset, FLAGS.dim, FLAGS.C))
  if FLAGS.run:
    outfile += '_' + FLAGS.run
  device = 'cuda'

  main_directory = FLAGS.datasets_dir
  main_directory = os.path.expanduser(main_directory)
  dataset_dir = os.path.join(main_directory, FLAGS.dataset)
  if not os.path.exists(dataset_dir):
    print('Dataset not found ' + FLAGS.dataset)
    print(', '.join(os.listdir(dataset_dir)))
    exit(-1)
  graph_file = os.path.join(dataset_dir, 'train.txt.npy')
  edges = np.load(graph_file)
  pyg_edges = np.concatenate([edges, edges[:, ::-1]], axis=0).T
  pyg_edges = torch.from_numpy( np.array(pyg_edges, dtype='int64'))
  test_neg_file = os.path.join(dataset_dir, 'test.neg.txt.npy')
  test_neg_arr = np.load(open(test_neg_file, 'rb'))
  test_pos_file = os.path.join(dataset_dir, 'test.txt.npy')
  test_pos_arr = np.load(open(test_pos_file, 'rb'))


  model = Node2Vec(pyg_edges, embedding_dim=FLAGS.dim, walk_length=FLAGS.C,
                  context_size=FLAGS.C, walks_per_node=20, num_negative_samples=1,
                  sparse=True).to(device)
  

  loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
  optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

  def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

  def test():
    model.eval()
    embeds = model()
    npe = embeds.detach().cpu().numpy()
    test_scores = (npe[test_pos_arr[:, 0]]  * npe[test_pos_arr[:, 1]]).sum(-1)
    test_neg_scores = (npe[test_neg_arr[:, 0]] * npe[test_neg_arr[:, 1]]).sum(-1) 

    test_y = [0] * len(test_neg_scores) + [1] * len(test_scores)
    test_y_pred = np.concatenate([test_neg_scores, test_scores], 0)
    test_accuracy = metrics.roc_auc_score(test_y, test_y_pred)
    return test_accuracy
  
  header = 'epoch,time,accuracy'
  with open(outfile, 'w') as fout:
    print('writing to ' + outfile)
    fout.write(header + '\n')
    print(header)
    for epoch in range(1, 100):  # Over 100, it starts overfitting.
      loss = train()
      acc = test()
      line = '%i,%f,%f' % (epoch, time.time() - start, acc)
      print(line)
      fout.write(line + '\n')
      #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
  

  

if __name__ == '__main__':
  app.run(main)

