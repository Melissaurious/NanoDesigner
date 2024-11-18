#!/bin/zsh


CODE_DIR=./NanoDesigner/dyMEAN
#######
# Notes from dyMEAN authors:
# TypeError: Descriptors cannot be created directly.
# If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
# If you cannot immediately regenerate your protos, some other possible workarounds are:
#  1. Downgrade the protobuf package to 3.20.x or lower.
#  2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
#######

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

source ~/.bashrc
conda activate nanodesigner1


if [ -z "$GPU" ]; then
    GPU=0
fi

echo "Using GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

# Define the input directory generated with /NanoDesigner/jupyter_notebooks/process_datasets.ipynb Notebook
FOLD_10_DIR=./NanoDesigner/built_datasets/Nanobody_clustered_CDRH3
OUT_DIR=./NanoDesigner/dyMEAN_Nanobody_clustered_CDRH3  # Define the output directory
echo "Creating $OUT_DIR"
mkdir -p $OUT_DIR



# Expected Input:

# ├── /built_datasets/Nanobody_clustered_CDRH3
# │   ├── fold_0
# │   │   ├── train.json
# │   │   └── valid.json
# │   │   └── test.json
# │   │   ...
# │   ├── fold_1
# │   └── fold_9



# Copy the subfolders and contents from FOLD_10_DIR to OUT_DIR
rsync -av --progress "${FOLD_10_DIR}/" "${OUT_DIR}/"


# I have 10 folds, from 0 to 9
n=10
for ((i=0; i<=n; i++)); do

    echo "Processing fold $i, generating conserved template"

    # Construct paths
    template_file=${OUT_DIR}/fold_${i}/template.json
    train_dataset=${OUT_DIR}/fold_${i}/train.json
    val_dataset=${OUT_DIR}/fold_${i}/valid.json

    python ${CODE_DIR}/data/framework_templates.py \
        --dataset ${train_dataset} \
        --out ${template_file}

    # Start training
    echo "Training fold $i"
    cd $CODE_DIR
    python ${CODE_DIR}/train.py \
        --gpu $GPU \
        --train_set ${train_dataset} \
        --valid_set ${val_dataset} \
        --template_path ${template_file} \
        --cdr "H3" \
        --max_epoch 200 \
        --save_topk 10 \
        --batch_size 16 \
        --shuffle \
        --model_type "dyMEAN" \
        --embed_dim 64 \
        --hidden_size 128 \
        --k_neighbors 9 \
        --n_layers 3 \
        --iter_round 3 \
        --bind_dist_cutoff 6.6

    echo "Creating empty folder to manually locate the best checkpoint based on the generated topk_map.txt file"
     
done


