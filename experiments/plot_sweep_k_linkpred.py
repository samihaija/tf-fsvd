
import csv
import collections


from absl import app, flags
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

flags.DEFINE_string('input', 'results/linkpred_d_sweep/fsvd.csv',
                    'Path to input CSV file.')
flags.DEFINE_string('output', 'results/sensitivity_k_linkpred.pdf',
                    'Path to write chart (should be PDF or image type)')
FLAGS = flags.FLAGS

MODELS = {
  'fsvd': ['#0068a8', ],
  'fsvd100': ['#00d5ff', ],
  'netmf_approx': ['#ff00f2'],
  'netmf_exact': ['#8a00b0'],
  'wys': ['#15b000'],
  'n2v': ['#b00000'],
}

RENAMES = {
  'soc-facebook': 'Facebook',
  'ppi': 'PPI',
  'ca-HepTh': 'HepTh',
  'ca-AstroPh': 'AstroPh',

  'wys': 'WYS',
  'fsvd': 'fSVD($k$=32)',
  'fsvd100': 'fSVD($k$=100)',
  'netmf_approx': 'NetMF',
  'netmf_exact':  '$\\widetilde{NetMF}$'
}

def plot_linkpred_ours(csv_file, output_file):
  dataset_info = {
    'ppi': ('PPI', 'g--'),
    'ca-AstroPh': ('AstroPh', 'r-'),
    'ca-HepTh': ('HepTh', ''),
    'soc-facebook': ('Facebook', 'b-'),
  }

  fig = plt.figure(figsize=(5, 3))
  ax = plt.gca()
  plt.grid(True, which='both')

  dataset2depth2accuracy = collections.defaultdict(lambda: collections.defaultdict(list))

  by_dataset = {}
  for record in csv.DictReader(open(csv_file)):
    dataset = record['dataset'] 
    if dataset not in by_dataset:
      by_dataset[dataset] = {'x': [], 'y': []}
    by_dataset[dataset]['x'].append(int(record['dim'])//2)
    by_dataset[dataset]['y'].append(float(record['test']))
  
  for dataset, metrics in sorted(by_dataset.items()):
    X = metrics['x']
    Y = metrics['y']
    if dataset in dataset_info:
      dataset = dataset_info[dataset][0]
    ax.plot(X, Y, label=dataset, linestyle='--', marker='o', markersize=3)
    
  ax.set_xticks(range(2, 33, 2))
  
  ax.legend()
  plt.ylabel('Test ROC AUC', fontsize=12)
  plt.xlabel('fSVD Rank (=$k$)', fontsize=12)
  plt.tight_layout()
  plt.savefig(output_file)
  print('wrote ' + output_file)


def main(_):
  plot_linkpred_ours(FLAGS.input, FLAGS.output)


if __name__ == '__main__':
  app.run(main)
