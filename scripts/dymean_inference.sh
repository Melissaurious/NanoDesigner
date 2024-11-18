#!/bin/bash


echo "Make sure the best checkpoint for each is selected and renamed before running this script as:"
echo "/inference_best_ckpt/best.ckpt"
CODE_DIR=./NanoDesigner/dyMEAN
cd $CODE_DIR
MODEL=dyMEAN
MAIN_DIR=./NanoDesigner/dyMEAN_Nanobody_clustered_CDRH3
TOT_FOLDS=10



# Source .bashrc to ensure Conda is initialized
source ~/.bashrc
conda activate nanodesigner1

# module load greasy

if [ -z ${GPU} ]; then
    GPU=0
fi


echo --------- Running Inference ----------------

# Iterate over each fold
f=$TOT_FOLDS
for ((i=0; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    # Define test set path
    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json
    TEMPLATE=${MAIN_DIR}/fold_${FOLD}/template.json

    # Create a results directory
    FOLD_RESULTS_DIR=${MAIN_DIR}/fold_${FOLD}/results
    mkdir -p ${FOLD_RESULTS_DIR}


    # Define output directory
    OUT_DIR=${FOLD_RESULTS_DIR}

    #Define Path to checkpoint
    CKPT_FILE=${MAIN_DIR}/fold_${FOLD}/inference_best_ckpt/best.ckpt

    #Run script
    python ${CODE_DIR}/generate_modified.py \
        --ckpt ${CKPT_FILE} \
        --test_set ${TEST_SET} \
        --save_dir ${FOLD_RESULTS_DIR} \
        --template ${TEMPLATE} \
        --gpu 0

    wait

done


echo --------- Conducting Refinement --------------

# Iterate over each fold
f=$TOT_FOLDS
for ((i=0; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"
    echo "Skipping Side Chain PAcking as dyMEAN is end to end... Refining ${FOLD}"

    # Define test set path
    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json
 
    #define summary file
    SUMMARY_INFERENCE=${FOLD_RESULTS_DIR}/summary.json
    SUMMARY_PACKED=${FOLD_RESULTS_DIR}/summary_refined.json

    python ${CODE_DIR}/models/pipeline/NanoDesigner_refinement_spacked_complexes.py \
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
for ((i=0; i<=f; i++)); do

    FOLD=$i
    echo "Processing fold ${FOLD}"

    TEST_SET=${MAIN_DIR}/fold_${FOLD}/test.json
    FOLD_RESULTS_DIR=${MAIN_DIR}/fold_${i}/results
    SUMMARY_FILE=${FOLD_RESULTS_DIR}/summary_refined.json
    METRICS_FILE=${METRICS_DIR}/metrics_fold_${FOLD}.json


    # #Run script
    python ${CODE_DIR}/cal_metrics_modified.py \
        --summary_json ${SUMMARY_FILE} \
        --metrics_file ${METRICS_FILE} \
        --cdr_model ${MODEL} \
        --cdr_type H3 \
        --test_set ${TEST_SET}
    
    wait

done


echo "Results are located at ${METRICS_DIR}"