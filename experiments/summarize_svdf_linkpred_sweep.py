import os
import glob
import collections
import json
import numpy as np

from absl import app, flags

flags.DEFINE_string('results_dir', 'results/linkpred_d_sweep/fsvd',
                    'Directory where run files are written.')
FLAGS = flags.FLAGS

def main(_):
  files = glob.glob(os.path.join(FLAGS.results_dir, '*'))
  stats = collections.defaultdict(list)
  for fname in files:
    #print(fname)
    dataset, d, run_id = fname.split('/')[-1].replace('.txt', '').split('_')
    d = int(d)
    lines = open(fname).read().split('\n')
    if not lines[-1]: lines=lines[:-1] # Remove last line (if blank)
    data = json.loads(lines[-1])
    #print(data)
    stats[(dataset, d)].append((data['auc'], data['time']))

  print('model,dataset,dim,test,time')
  for k in list(sorted(stats.keys())):
    stats[k] = np.array(stats[k])
    dataset, d = k
    # Total embedding dimension is twice the rank, as node is embedded in U and V.
    d *= 2
    print('fsvd,%s,%i,%g,%g' % (
        dataset, d, np.mean(stats[k][:, 0]), np.mean(stats[k][:, 1])))

if __name__ == '__main__':
  app.run(main)
