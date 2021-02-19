PROGRAM='python3 baselines/netmf.py'
OUT_MODELS_DIR='results/baselines/netmf/models'
mkdir -p ${OUT_MODELS_DIR}

D=64
# "large" means: run the "approximate" NetMF
SIZE=large

WINDOW=5
DS=ppi
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

DS=soc-facebook
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

WINDOW=20
DS=ca-AstroPh
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

DS=ca-HepTh
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt


# "small" means: run the "exact" NetMF
SIZE=small
WINDOW=5
DS=ppi
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

DS=soc-facebook
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

WINDOW=20
DS=ca-AstroPh
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt

DS=ca-HepTh
${PROGRAM} --${SIZE} --window=${WINDOW} --input datasets/asymproj/${DS}/train.txt.npy --rank=${D} --output ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}.npy > ${OUT_MODELS_DIR}/${DS}_${SIZE}_D${D}_W${WINDOW}_output.txt
