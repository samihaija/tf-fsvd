import os


OUTPUT_DIR = 'results/linkpred_d_sweep/fsvd'

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)


DATASETS = ['ca-AstroPh', 'ca-HepTh', 'soc-facebook', 'ppi']

for run in range(3):
  for dim in range(2, 33, 1):
    for dataset in DATASETS:
      output_file = os.path.join(OUTPUT_DIR, '%s_%i_%s' % (dataset, dim, str(run)))
      print('python3.8 implementations/linkpred_asymproj.py --window=5 --neg_coef=3 --dataset_name=%s --dim=%i > %s' % (
        dataset, dim, output_file
      ))
