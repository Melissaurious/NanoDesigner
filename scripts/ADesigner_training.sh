#!/bin/zsh
CODE_DIR=./NanoDesigner/ADesigner/

#######
# TypeError: Descriptors cannot be created directly.
# If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
# If you cannot immediately regenerate your protos, some other possible workarounds are:
#  1. Downgrade the protobuf package to 3.20.x or lower.
#  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
#######

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

source ~/.bashrc
conda activate dymean0


if [ -z "$GPU" ]; then
    GPU=0
fi

echo "Using GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU


# Define the input directory generated with /NanoDesigner/jupyter_notebooks/process_datasets.ipynb Notebook
FOLD_10_DIR=./NanoDesigner/built_datasets/Nanobody_clustered_CDRH3
DIR=./NanoDesigner/ADesigner_Nanobody_clustered_CDRH3

echo "Creating $DIR"
mkdir -p $DIR

# Copy the subfolders and contents from FOLD_10_DIR to OUT_DIR
rsync -av --progress "${FOLD_10_DIR}/" "${DIR}/"


# Define n= number of folds you have
n=10
for ((i=0; i<=n; i++)); do

    echo "Training fold $i"

    # Construct paths
    train_dataset=${DIR}/fold_${i}/train.json
    val_dataset=${DIR}/fold_${i}/valid.json
    save_dir=${DIR}/fold_${i}/processed_entries
    fold=${i}

    # Start training
    cd $CODE_DIR
    python ${CODE_DIR}/ADesigner/trainer.py \
        --train_set ${train_dataset} \
        --valid_set ${val_dataset} \
        --save_dir ${save_dir} \
        --fold ${i}

done