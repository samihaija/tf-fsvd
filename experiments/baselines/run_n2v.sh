export CUDA_VISIBLE_DEVICES=1
PROGRAM='python3 baselines/pyg_node2vec.py'
${PROGRAM} --dataset=ca-HepTh --C=20 
${PROGRAM} --dataset=ca-AstroPh --C=20
${PROGRAM} --dataset=ppi
${PROGRAM} --dataset=soc-facebook
