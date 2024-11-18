#!/bin/bash

echo "Double check: the checkpoints are saved as:"
echo "/inference_best_ckpt/best.ckpt"
CODE_DIR=./NanoDesigner/diffab
cd $CODE_DIR
CODE_DIR_dyMEAN=./NanoDesigner/dyMEAN
MODEL=DiffAb
MAIN_DIR=./NanoDesigner/DiffAb_Nanobody_clustered_CDRH3
TOT_FOLDS=10


# Source .bashrc to ensure Conda is initialized
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanodesigner2

# source ~/.bashrc
# conda activate nanodesigner2

if [ -z ${GPU} ]; then
    GPU=0
fi

echo --------- Running Inference ----------------


# Iterate over each fold
f=$TOT_FOLDS
for ((i=1; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    # Define test set path
    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json

    # Create a results directory
    FOLD_RESULTS_DIR=${MAIN_DIR}/fold_${FOLD}/results
    mkdir -p ${FOLD_RESULTS_DIR}


    # Define output directory
    OUT_DIR=${FOLD_RESULTS_DIR}

    #Define Path to checkpoint
    CKPT_FOLDER=${MAIN_DIR}/fold_${FOLD}/checkpoints

    #Run script
    python ${CODE_DIR}/EVAL_CKPTS/design_for_pdb_singlecdr.py \
        --ckpt_folder ${CKPT_FOLDER} \
        --test_set ${TEST_SET} \
        --out_dir ${FOLD_RESULTS_DIR} \
        --FOLD ${FOLD} \
        --dymean_code ${CODE_DIR_dyMEAN} \
        --num_samples 1

    wait

done

conda deactivate

conda activate nanodesigner1
cd $CODE_DIR_dyMEAN

echo --------- Conducting Side Chain Packing --------------

# Iterate over each fold
f=$TOT_FOLDS
for ((i=1; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    # Define test set path
    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json

    #define summary file
    SUMMARY_INFERENCE=${FOLD_RESULTS_DIR}/summary.json
    SUMMARY_PACKED=${FOLD_RESULTS_DIR}/summary_packed.json

    cd $CODE_DIR_dyMEAN

    python ${CODE_DIR_dyMEAN}/models/pipeline/NanoDesigner_refinement_spacked_complexes.py \
        --in_file ${SUMMARY_INFERENCE} \
        --out_file ${SUMMARY_PACKED} \
        --cdr_model ${MODEL}
    wait



done


echo --------- Conducting Refinement after Sidechain packing --------------

# Iterate over each fold
f=$TOT_FOLDS
for ((i=1; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    # Define test set path
    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json
    cd $CODE_DIR_dyMEAN
 
    #define summary file
    SUMMARY_PACKED=${FOLD_RESULTS_DIR}/summary_packed.json
    SUMMARY_PACKED_and_REFINED=${FOLD_RESULTS_DIR}/summary_packed_and_refined.json

    python ${CODE_DIR_dyMEAN}/models/pipeline/NanoDesigner_refinement_spacked_complexes.py \
        --in_file ${SUMMARY_INFERENCE} \
        --out_file ${SUMMARY_PACKED} \
        --cdr_model ${MODEL}
    wait

done


echo --------- Conducting Evaluation ---------------

METRICS_DIR=${MAIN_DIR}/metrics
mkdir -p ${METRICS_DIR}

# Iterate over each fold
f=$TOT_FOLDS
for ((i=1; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json
    FOLD_RESULTS_DIR=${MAIN_DIR}/fold_${i}/results
    SUMMARY_FILE=${FOLD_RESULTS_DIR}/summary_packed_and_refined.json
    METRICS_FILE=${METRICS_DIR}/metrics_fold_${FOLD}.json

    cd $CODE_DIR_dyMEAN

    # #Run script
    python ${CODE_DIR_dyMEAN}/cal_metrics_modified.py \
        --summary_json ${SUMMARY_FILE} \
        --metrics_file ${METRICS_FILE} \
        --cdr_model ${MODEL} \
        --cdr_type H3 \
        --test_set ${TEST_SET}
    
    wait

done
