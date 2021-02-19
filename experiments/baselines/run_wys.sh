
mkdir -p results/baselines/wys

PROGRAM='python3.8 baselines/wys.py'

D=32
DS=ppi
date > results/baselines/wys/${DS}_${D}.txt
${PROGRAM} --dataset_dir datasets/asymproj/${DS} --d ${D} 2>> results/baselines/wys/${DS}_${D}.txt
date >> results/baselines/wys/${DS}_${D}.txt

DS='soc-facebook'
date > results/baselines/wys/${DS}_${D}.txt
${PROGRAM} --dataset_dir datasets/asymproj/${DS} --d ${D} 2>> results/baselines/wys/${DS}_${D}.txt
date >> results/baselines/wys/${DS}_${D}.txt

DS='ca-HepTh'
date > results/baselines/wys/${DS}_${D}.txt
${PROGRAM} --dataset_dir datasets/asymproj/${DS} --d ${D} 2>> results/baselines/wys/${DS}_${D}.txt
date >> results/baselines/wys/${DS}_${D}.txt

DS='ca-AstroPh'
date > results/baselines/wys/${DS}_${D}.txt
${PROGRAM} --dataset_dir datasets/asymproj/${DS} --d ${D} 2>> results/baselines/wys/${DS}_${D}.txt
date >> results/baselines/wys/${DS}_${D}.txt

