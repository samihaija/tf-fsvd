
import csv
import glob 
import json
import collections

from absl import app, flags
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np



def plot_classification_ours(globs, output_file):
  dataset_info = {
    'ind.cora': ('Cora', 'g--'),
    'ind.citeseer': ('Citeseer', 'r-'),
    'ind.pubmed': ('Pubmed', 'b-'),
  }

  fig = plt.figure(figsize=(5, 3))
  ax = plt.gca()
  #ax.set_xscale('log')
  plt.grid(True, which='both')

  dataset2depth2accuracy = collections.defaultdict(lambda: collections.defaultdict(list))

  for g in globs:
    allfiles = glob.glob(g)
    
    for filename in allfiles:
      print(filename)
      specs = [tuple(part.split('.', 1)) for part in filename.split('/')[-1][:-5].split('_')]
      
      specs = [s for s in specs if len(s)==2]
      specs = dict(specs)
      results = json.loads(open(filename).read())
      test_accuracy = results['test']
      if int(specs['layers']) > 15: continue
      dataset2depth2accuracy[specs['dataset']][int(specs['layers'])].append(test_accuracy)

  for dataset in sorted(dataset2depth2accuracy.keys()):
    accuracy_by_layers = dataset2depth2accuracy[dataset]
    X = []
    Y = []
    YERR = []
    label, linestyle = dataset_info[dataset]
    
    for layers, accuracies in sorted(accuracy_by_layers.items()):
      X.append(layers)
      Y.append(np.mean(accuracies))
      YERR.append(np.std(accuracies))
    #ax.plot(X, Y, )
    ax.errorbar(X, Y, yerr=YERR,  label=label, linestyle='--', marker='o', markersize=3)
    
  ax.set_xticks(range(16))
  ax.set_yticks([0.5, 0.6, 0.7, 0.8])
  
  ax.legend()
  plt.ylabel('Test Accuracy', fontsize=12)
  plt.xlabel('Num Layers (=$L$)', fontsize=12)
  plt.tight_layout()
  #import IPython; IPython.embed()
  plt.savefig(output_file)
  print('wrote ' + output_file)
  





def main(_):

  plot_classification_ours(
    [
    'results/planetoid_hp_search/*_wys_window.1_wys_neg_coef.1*.json'
    ], 'results/sensitivity_planetoid.pdf'
  )


if __name__ == '__main__':
  app.run(main)
