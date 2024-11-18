#!/bin/zsh
#Run DiffAb training fold by fold

# Define variables
CODE_DIR=./NanoDesigner/diffab
cd $CODE_DIR
CONFIG=${CODE_DIR}/configs/train/codesign_single.yml #UDPDATE required content
FOLD=1

# Define the input directory generated with /NanoDesigner/jupyter_notebooks/process_datasets.ipynb Notebook
FOLD_10_DIR=./NanoDesigner/built_datasets/Nanobody_clustered_CDRH3
DIR=./NanoDesigner/DiffAb_Nanobody_clustered_CDRH3 # save trained models


"""
Note: prior training create chothia_dir using 
./NanoDesigner/jupyter_notebooks/process_datasets.ipynb
Chothia for Nanobodies
Chothia for antibodies and merger the folders into one, update the CONFIG accordingly
"""

# Source .bashrc to ensure Conda is initialized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanodesigner2

# source ~/.bashrc
# conda activate nanodesigner2


#Create path to training and validation data
val_json=${FOLD_10_DIR}/fold_${FOLD}/valid.json
train_json=${FOLD_10_DIR}/fold_${FOLD}/train.json


python ${CODE_DIR}/train_modified.py \
    --config ${CONFIG} \
    --fold ${FOLD} \
    --valid_json_file ${val_json} \
    --train_json_file ${train_json} \
    --out_folder ${DIR} \
    --special_filter True
    #--resume \ 
    # --tag  \
    # --finetune \
    # --debug  \
    # --device  \
    # --num_workers  \
