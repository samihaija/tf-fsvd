"""Runs a grid-search on hyperparmeters for training on the planetoid dataset."""

import itertools
import tqdm
import subprocess
import os

from absl import flags, app

flags.DEFINE_string('output_dir', 'results/planetoid_hp_search',
                    'Directory will be created and json files will be written')
flags.DEFINE_string('wys_windows', '1,2,3,4,5',
                    'Comma-separated list of WYS windows')
flags.DEFINE_string('wys_neg_coefs', '1,2,3',
                    'Comma-separated list of negative coefficients (lambda)')
flags.DEFINE_string('datasets', 'ind.cora,ind.pubmed,ind.citeseer',
                    'comma-separated list of dataset names to run on')
flags.DEFINE_string('layers', '2,4,8,16',
                    'Comma-separated list of layers to try')
flags.DEFINE_string('run_ids', '0,1,2',
                    'Comma-separated RUN IDs')
FLAGS = flags.FLAGS

def main(_):
  RESULTS_DIR = FLAGS.output_dir
  PYTHON = os.environ.get('PYTHON', 'python3')

  if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

  WYS_WINDOW = [int(i) for i in FLAGS.wys_windows.split(',')]
  WYS_NEG_COEF = [int(i) for i in FLAGS.wys_neg_coefs.split(',')]
  LAYERS = [int(i) for i in FLAGS.layers.split(',')]
  RUN = FLAGS.run_ids.split(',')

  DATASET = ['ind.cora', 'ind.pubmed', 'ind.citeseer']

  experiments_prod = list(
      itertools.product(DATASET, LAYERS, WYS_WINDOW, WYS_NEG_COEF, RUN))
  for (dataset, layers, wys_window, wys_neg_coef, run) in tqdm.tqdm(experiments_prod):
    args = [('dataset', dataset), ('layers', layers),
            ('wys_window', wys_window), ('wys_neg_coef', wys_neg_coef)]
    
    filename =  '_'.join(['%s.%s' % (argname, str(argval)) for (argname, argval) in args])
    filename = '%s_run.%s.json' % (filename, str(run))
    out_file = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(out_file):
      print('Skipping already-computed: %s' % out_file)
    cmd = ([PYTHON, 'implementations/node_ssc_planetoid.py'] +
           ['--%s=%s' % (argname, str(argval)) for (argname, argval) in args])
    run_out = subprocess.check_output(cmd)
    result_json = '\n'.join(run_out.decode().split('\n')[-2:])
    with open(out_file, 'w') as fout:
      fout.write(result_json)

if __name__ == '__main__':
  app.run(main)
