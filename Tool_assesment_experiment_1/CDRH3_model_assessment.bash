#!/bin/bash

# ===== GPU CONFIGURATION =====
# Assign one GPU per job (0-3 for 4 GPUs)
GPU_ID=0  # Get GPU ID from first script argument
export CUDA_VISIBLE_DEVICES=$GPU_ID

eval "$(conda shell.bash hook)"

# ===== START OF CONFIGURATION =====
MODEL="DIFFAB" # "dyMEAN" or "ADESIGN" or "DIFFAB"
BASE_DIR="./"
PROJECT_DIR="${BASE_DIR}/NanoDesigner" 
DYMEAN_CODE_DIR="${PROJECT_DIR}/dyMEAN"
ADESIGNER_CODE_DIR="${PROJECT_DIR}/ADesigner"
DIFFAB_CODE_DIR="${PROJECT_DIR}/diffab"
cd ${DYMEAN_CODE_DIR}
DATA_DIR="${PROJECT_DIR}/Data_download_and_processing"
OUTPUT_ROOT="${PROJECT_DIR}/Tool_assesment_experiment_1/${MODEL}_ASSESMENT"
TRAIN_FOLDER="${PROJECT_DIR}/all_checkpoints/${MODEL}"


# Select clustering configuration (modify this for each job)
CLUSTER_TYPE="Ag"       # "Ag" or "CDRH3"
CLUSTER_THRESHOLD="60"   # "60", "80", "95",  for antigen; for CDRH3 is 40, 30, 20
IMMUNO_MOLECULE="Nanobody_Antibody" # or  Nanobody, when finetunning TRUE, immuno molecules will be set autoamtically
TOTAL_FOLDS=9

# Configuration variable - THIS CONTROLS EVERYTHING
FINETUNE=TRUE  # Set to TRUE to search for fine-tuned checkpoints

# ===== END OF CONFIGURATION =====
export DIFFAB_CODE_DIR="${PROJECT_DIR}/diffab"
export DYMEAN_CODE_DIR="${PROJECT_DIR}/dyMEAN"
export ADESIGNER_CODE_DIR="${PROJECT_DIR}/ADesigner"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export NANOBODIES_PROJECT_DIR="${PROJECT_DIR}"




# For fine-tuning, switch to Nanobody data
if [ "$FINETUNE" == "TRUE" ]; then
    IMMUNO_MOLECULE="Nanobody"
    FINETUNE_SOURCE="Antibody"  # Original model was trained on Antibodies
else
    IMMUNO_MOLECULE=${IMMUNO_MOLECULE}
fi


# # Iterate over each fold
for (( FOLD=0; FOLD<=TOTAL_FOLDS; FOLD++ )); do

    # ===== DERIVED PATHS =====
    if [ "$FINETUNE" == "TRUE" ]; then

        # For fine-tuning: use Nanobody data but save in Antibody directory with _fine_tuned suffix
        CLUSTER_DIR="${IMMUNO_MOLECULE}_clustered_${CLUSTER_TYPE}_${CLUSTER_THRESHOLD}"
        INPUT_DATA_DIR="${DATA_DIR}/${CLUSTER_DIR}"
        
        # Save fine-tuned model in Antibody directory with clear distinction
        SOURCE_CLUSTER_DIR="${FINETUNE_SOURCE}_clustered_${CLUSTER_TYPE}_${CLUSTER_THRESHOLD}"
        OUTPUT_DIR="${OUTPUT_ROOT}/${SOURCE_CLUSTER_DIR}"  # Same as source but different fold name

        SOURCE_CKPT_DIR="${TRAIN_FOLDER}/${SOURCE_CLUSTER_DIR}/fold_${FOLD}_fine_tuned/"
        
    else
        CLUSTER_DIR="${IMMUNO_MOLECULE}_clustered_${CLUSTER_TYPE}_${CLUSTER_THRESHOLD}"
        INPUT_DATA_DIR="${DATA_DIR}/${CLUSTER_DIR}"
        SOURCE_CLUSTER_DIR="${IMMUNO_MOLECULE}_clustered_${CLUSTER_TYPE}_${CLUSTER_THRESHOLD}"
        OUTPUT_DIR="${OUTPUT_ROOT}/${CLUSTER_DIR}"

        SOURCE_CKPT_DIR="${TRAIN_FOLDER}/${SOURCE_CLUSTER_DIR}/fold_${FOLD}"
    fi

    METRICS_DIR="${OUTPUT_DIR}/metrics_per_fold/"
    mkdir -p "${METRICS_DIR}"
    echo "${METRICS_DIR}"

    TEST_FILE="${INPUT_DATA_DIR}/fold_${FOLD}/test.json"
    echo "${TEST_FILE}"

    # # Create output directories
    FOLD_OUTPUT="${OUTPUT_DIR}/fold_${FOLD}"
    echo "${FOLD_OUTPUT}"
    mkdir -p "${FOLD_OUTPUT}"
    cp "${TEST_FILE}" "${FOLD_OUTPUT}/"


    TEST_FILE="${FOLD_OUTPUT}/test.json"
    echo "${TEST_FILE}"

    RESULTS_DIR="${FOLD_OUTPUT}/results"
    mkdir -p "${RESULTS_DIR}"
    echo "${RESULTS_DIR}"

    # LOCATE CKPTS - model specific

    if [ "$MODEL" = "DIFFAB" ]; then
        # Default checkpoint (corrected path)

        if [ "$FINETUNE" == "TRUE" ]; then
            CKPT="${SOURCE_CKPT_DIR}/30000.pt"
        else
            CKPT="${SOURCE_CKPT_DIR}/200000.pt"
        fi
        
        # If default doesn't exist, find the latest checkpoint
        if [ ! -f "$CKPT" ]; then
            echo "Default checkpoint not found: $CKPT" >&2
            
            # Check if checkpoints directory exists
            if [ ! -d "${SOURCE_CKPT_DIR}/checkpoints" ]; then
                echo "Error: Checkpoints directory not found: ${SOURCE_CKPT_DIR}/" >&2
                # exit 1
                continue 
            fi
            
            # Find all checkpoint files and sort numerically, then take the last one
            LATEST_CKPT=$(ls ${SOURCE_CKPT_DIR}/*.pt 2>/dev/null | sort -V | tail -n 1)
            
            if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
                CKPT="$LATEST_CKPT"
                echo "Using latest checkpoint: $CKPT" >&2
            else
                echo "Error: No checkpoint files found in ${SOURCE_CKPT_DIR}/" >&2
                # exit 1
                continue 
            fi
        else
            echo "Using default checkpoint: $CKPT" >&2
        fi
        
        # Verify checkpoint file exists before proceeding
        if [ ! -f "$CKPT" ]; then
            echo "Error: Checkpoint file does not exist: $CKPT" >&2
            # exit 1
            continue 
        fi
        

        conda activate nanodesigner2
        
        # Verify conda environment activated successfully
        if [ $? -ne 0 ]; then
            echo "Error: Failed to activate diffab conda environment" >&2
            # exit 1
            continue 
        fi
        
        echo "Running DIFFAB inference with checkpoint: $CKPT" >&2
        
        cd ${DIFFAB_CODE_DIR}
        python "${DIFFAB_CODE_DIR}/EVAL_CKPTS/design_for_pdb_2025.py" \
            --checkpoint "${CKPT}" \
            --test_set "${TEST_FILE}" \
            --out_dir "${RESULTS_DIR}" \
            --num_samples 1 \
            --filter_2_Ag_entries \
            --max_num_test_entries 100 \
            --design_mode  single_cdr
        
        # Capture exit status
        PYTHON_EXIT_STATUS=$?
        
        # Wait for any background processes (though none should be running)
        wait
        
        # Deactivate environment
        conda deactivate
        
        # Check if Python script succeeded
        if [ $PYTHON_EXIT_STATUS -ne 0 ]; then
            echo "Error: Python script failed with exit status $PYTHON_EXIT_STATUS" >&2
            # exit $PYTHON_EXIT_STATUS
            continue 
        fi
        
        echo "DIFFAB inference completed successfully" >&2
    # fi

    # LOCATE CKPTS - model specific
    elif [ "$MODEL" = "dyMEAN" ]; then
        # Find all version directories and sort them to get the latest one
        LATEST_VERSION_DIR=$(find "${SOURCE_CKPT_DIR}" -maxdepth 1 -type d -name "version_*" | sort -V | tail -n 1)
        
        if [ -z "$LATEST_VERSION_DIR" ]; then
            echo "Error: No version directories found in $SOURCE_CKPT_DIR" >&2
            continue 
        fi

        TOPK_FILE="${LATEST_VERSION_DIR}/checkpoint/topk_map.txt"
        
        if [ ! -f "$TOPK_FILE" ]; then
            echo "Error: topk_map.txt not found in $LATEST_VERSION_DIR/checkpoint/" >&2
            continue 
        fi

        # Extract the first line and get the path after the colon
        OLD_CKPT=$(head -n 1 "$TOPK_FILE" | awk -F': ' '{print $2}')
        
        if [ -z "$OLD_CKPT" ]; then
            echo "Error: Could not extract checkpoint path from topk_map.txt" >&2
            continue 
        fi

        # Replace old path with new path
        OLD_TRAIN_DIR="/home/rioszemm/data/dyMEAN_TRAIN_MAY_2025"
        CKPT="${OLD_CKPT/$OLD_TRAIN_DIR/$TRAIN_FOLDER}"

        echo "Original checkpoint path: $OLD_CKPT" >&2
        echo "Using dyMEAN checkpoint: $CKPT" >&2

        conda activate nanodesigner1

        cd ${DYMEAN_CODE_DIR}
        TEMPLATE="${SOURCE_CKPT_DIR}/template.json"  # if finetuned, copied the one for antibody into the new fold_X_fintuned folder.
        echo "${TEMPLATE}"
        python /ibex/user/rioszemm/NanobodiesProject/dyMEAN/EVAL_CKPTS/generate_june_2025.py \
            --ckpt ${CKPT} \
            --test_set ${TEST_FILE} \
            --save_dir ${RESULTS_DIR} \
            --template ${TEMPLATE} \
            --filter_2_Ag_entries \
            --max_num_test_entries 100 \
            --gpu ${GPU_ID}

    elif [ "$MODEL" = "ADESIGN" ]; then

        CKPT="${SOURCE_CKPT_DIR}/best.ckpt"


        if [ ! -f "$CKPT" ]; then
            echo "Error: ADESIGN checkpoint not found at $CKPT" >&2
            # exit 1
            continue 
        fi
        
        echo "Using ADESIGN checkpoint: $CKPT" >&2

        # source ~/.bashrc
        conda activate nanodesigner1


        
        PROCESSED_ENTRIES="${SOURCE_CKPT_DIR}/part_0.pkl"
        echo "${PROCESSED_ENTRIES}"
        cd ${ADESIGNER_CODE_DIR}

        python ${ADESIGNER_CODE_DIR}/generate_assessment_2025.py \
            --ckpt ${CKPT} \
            --test_set ${TEST_FILE} \
            --save_dir ${RESULTS_DIR} \
            --preprocessed_path ${PROCESSED_ENTRIES} \
            --filter_2_Ag_entries \
            --max_num_test_entries 100 \
            --gpu ${GPU_ID}


    else
        echo "Error: Unknown model type $MODEL" >&2
        # exit 1
        continue 
    fi

    cd ${DYMEAN_CODE_DIR}

    SUMMARY_INFERENCE=${RESULTS_DIR}/summary.json
    SUMMARY_PACKED=${RESULTS_DIR}/summary_packed.json
    SUMMARY_FILE_PACKED_REFINED=${RESULTS_DIR}/summary_packed_and_refined.json

    echo "Performing Side chain Packing"

    conda activate nanodesigner1

    if [ "$MODEL" = "ADESIGN" ] || [ "$MODEL" = "DIFFAB" ]; then
        
        python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_side_chain_packing.py \
            --summary_json ${SUMMARY_INFERENCE} \
            --out_file ${SUMMARY_PACKED} \
            --test_set ${TEST_FILE} \
            --cdr_model ${MODEL}

        wait
        
        # For ADESIGN and DIFFAB, use the packed summary as input for refinement
        REFINEMENT_INPUT=${SUMMARY_PACKED}
    else
        # For dyMEAN, skip packing and use the original summary as input for refinement
        REFINEMENT_INPUT=${SUMMARY_INFERENCE}
    fi



    echo --------- Refinement---------------


    start_time=$SECONDS
    python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_refinement_after_side_chain_packing_parallel.py \
        --cdr_model ${MODEL} \
        --in_file "${REFINEMENT_INPUT}" \
        --out_file "${SUMMARY_FILE_PACKED_REFINED}"

    wait

    end_time=$SECONDS
    execution_time=$((end_time - start_time))
    echo "Refinement after Side Chain Packing took approximately: $execution_time seconds"




    echo "Performing Model Evaluation"

    METRICS_FILE=${METRICS_DIR}metrics_fold_${FOLD}.json

    echo "${METRICS_FILE}"

    python ${DYMEAN_CODE_DIR}/NanoDesigner_metrics_computations.py \
        --summary_json ${SUMMARY_FILE_PACKED_REFINED} \
        --metrics_file ${METRICS_FILE} \
        --cdr_model ${MODEL} \
        --cdr_type H3 H2 H1 \
        --test_set ${TEST_FILE}

    wait

done