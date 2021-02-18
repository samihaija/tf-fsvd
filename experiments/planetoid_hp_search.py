"""Searches through the hyperparmeters on the planetoid dataset."""

import itertools
import tqdm
import subprocess
import os

RESULTS_DIR = 'results/planetoid_hp_search'
PYTHON = os.environ.get('PYTHON', 'python3')

if not os.path.exists(RESULTS_DIR):
  os.makedirs(RESULTS_DIR)


WYS_WINDOW = [1, 2, 3, 4, 5]
WYS_NEG_COEF = [1, 2, 3]
LAYERS = range(17)
RUN=[0,1]

DATASET = ['ind.cora', 'ind.pubmed', 'ind.citeseer']

experiments_prod = list(itertools.product(DATASET, LAYERS, WYS_WINDOW, WYS_NEG_COEF, RUN))
for (dataset, layers, wys_window, wys_neg_coef, run) in tqdm.tqdm(experiments_prod):
  args = [('dataset', dataset), ('layers', layers), ('wys_window', wys_window), ('wys_neg_coef', wys_neg_coef)]
  
  cmd = [PYTHON, 'implementations/node_ssc_planetoid.py'] + ['--%s=%s' % (argname, str(argval)) for (argname, argval) in args]
  run_out = subprocess.check_output(cmd)
  result_json = '\n'.join(run_out.decode().split('\n')[-2:])
  filename =  '_'.join(['%s.%s' % (argname, str(argval)) for (argname, argval) in args])
  filename = '%s_run.%s.json' % (filename, str(run))
  with open(os.path.join(RESULTS_DIR, filename), 'w') as fout:
    fout.write(result_json)
