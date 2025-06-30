#!/bin/bash

# Check if at least 1 PDB code is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 PDB_CODE [STARTING_ITERATION]"
    echo "If STARTING_ITERATION is not provided, defaults to 1"
    exit 1
fi


# Define variables
BASENAME=$1
START_ITER=${2:-1}  # Use second argument if provided, otherwise default to 1


BASE_DIR="./"
PROJECT_DIR="${BASE_DIR}/NanoDesigner" 
DYMEAN_CODE_DIR="${PROJECT_DIR}/dyMEAN"
ADESIGNER_CODE_DIR="${PROJECT_DIR}/ADesigner"
DIFFAB_CODE_DIR="${PROJECT_DIR}/diffab"
cd ${DYMEAN_CODE_DIR}

CONFIG="${PROJECT_DIR}/config_files/NanoDesigner_diffab_denovo_1CDR.yml"
MAIN_FOLDER="${PROJECT_DIR}/NanoDesigner_assessment_experiment_2/NanoDesigner_DiffAb_denovo"
DATA_DIR="${PROJECT_DIR}/Data_download_and_processing"

# Construct path to test.json for fold 0 of Nanobody_Antibody clustered at 60 (should match the training configuration of the config yml file)
CLUSTER_DIR="Nanobody_Antibody_clustered_Ag_60"
INPUT_DATA_DIR="${DATA_DIR}/${CLUSTER_DIR}"
TEST_FILE="${INPUT_DATA_DIR}/fold_0/test.json"
OUTPUT_DIR="${MAIN_FOLDER}/${BASENAME}"

# Create directory
mkdir -p "$OUTPUT_DIR"

DATASET="${OUTPUT_DIR}/${BASENAME}.json"

# Check if test.json exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found at $TEST_FILE"
    exit 1
fi

# Debug: Print constructed paths
echo "DEBUG: MAIN_FOLDER = $MAIN_FOLDER"
echo "DEBUG: OUTPUT_DIR = $OUTPUT_DIR"
echo "DEBUG: OUTPUT_FILE = $DATASET"



# Extract the FIRST line matching the basename and save to new JSON file
echo "Extracting first entry with basename '$BASENAME' from $TEST_FILE"

# Use jq if available for better JSON handling, otherwise fallback to grep
if command -v jq &> /dev/null; then
    # Use jq to properly filter JSON entries and get only the first match
    jq -c "select(.pdb == \"$BASENAME\")" "$TEST_FILE" | head -n 1 > "$DATASET"
else
    # Fallback to grep and get only the first match
    grep "\"pdb\": \"$BASENAME\"" "$TEST_FILE" | head -n 1 > "$DATASET"
fi

# Check if any entries were found
if [ ! -s "$DATASET" ]; then
    echo "No entries found with basename '$BASENAME'"
    if [ -f "$DATASET" ]; then
        rm "$DATASET"
    fi
    if [ -d "$DATASET" ]; then
        rmdir "$DATASET" 2>/dev/null
    fi
    exit 1
else
    echo "Found and saved first entry with basename '$BASENAME'"
    echo "Saved to: $OUTPUT_FILE"
    echo "This matches the expected dataset path: DATASET=\"\${MAIN_FOLDER}/\${BASENAME}/\${BASENAME}.json\""
fi



# #NanoDesigner variables
R=50 # Number of randomized nanobodies (Initialization step)
N=15 # Top best mutants to proceed with to subsequent iterations
d=100 # docked models to generate
n=5 # top docked models to feed to inference stage
max_iter=10



# Extract variables from config file
MODEL=$(sed -n 's/^model_name: //p' "$CONFIG") # options: ADesigner or DiffAb
CKPT=$(sed -n 's/^[ ]*checkpoint: //p' "$CONFIG")
MAX_OBJECTIVE=$(sed -n 's/^[ ]*maximization_objective: //p' "$CONFIG") # options: dg for de novo; ddg for optimization design
CDRS=$(sed -n 's/^[ ]*CDRS: //p' "$CONFIG")
INITIAL_CDR=$(sed -n 's/^[ ]*initial_cdr: //p' "$CONFIG")
DATA_DIR=${MAIN_FOLDER}/${BASENAME}
SAVE_DIR=${DATA_DIR}/${MODEL}_${INITIAL_CDR}_cdr
HDOCK_DIR=${SAVE_DIR}/HDOCK
RESULT_DIR=${SAVE_DIR}/results


eval "$(conda shell.bash hook)"
conda activate dymean2


# create folder and parent dir if needed
mkdir -p $DATA_DIR



export DIFFAB_CODE_DIR="${PROJECT_DIR}/diffab"
export DYMEAN_CODE_DIR="${PROJECT_DIR}/dyMEAN"
export ADESIGNER_CODE_DIR="${PROJECT_DIR}/ADesigner"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export NANOBODIES_PROJECT_DIR="${PROJECT_DIR}"



# for ((i=1; i<=max_iter; i++)); do
for ((i=START_ITER; i<=max_iter; i++)); do





    echo "Iteration $i"
    echo "------Running docking simulation (Hdock) --------"


    SUMMARY_FILE_INFERENCE=${RESULT_DIR}_iteration_$i/summary_iter_${i}.json
    SUMMARY_FILE_PACKED=${RESULT_DIR}_iteration_$i/summary_packed_iter_${i}.json
    SUMMARY_FILE_PACKED_REFINED=${RESULT_DIR}_iteration_$i/summary_packed_refined_iter_${i}.json

    cd ${DYMEAN_CODE_DIR}
    start_time=$SECONDS
    python -m models.pipeline.NanoDesigner_structure_pred_and_docking \
        --dataset_json ${DATASET} \
        --randomized $R \
        --best_mutants $N \
        --cdr_type ${CDRS} \
        --cdr_model ${MODEL} \
        --hdock_models ${HDOCK_DIR}_iter_$i \
        --n_docked_models $d \
        --iteration $i \
        --cdr_model ${MODEL} \
        --initial_cdr ${INITIAL_CDR} \
        --csv_dir ${DATA_DIR}/csv_iter_$i \
        --csv_dir_ ${DATA_DIR}/csv_iter_ \


    wait
    end_time=$SECONDS
    execution_time=$(($end_time - $start_time))
    echo "Structure Prediction and Docking Simulation took approximately: $execution_time seconds"


    echo "------ Processing and selection of top docked Models --------"
    start_time=$SECONDS

    # Single run - no loop needed, Python script handles everything
    python -m models.pipeline.NanoDesigner_select_top_docked_models \
        --test_set ${DATASET} \
        --hdock_models ${HDOCK_DIR}_iter_$i \
        --iteration $i \
        --top_n $n

    # Calculate execution time
    end_time=$SECONDS
    execution_time=$((end_time - start_time))
    echo "Selection of top docked models took approximately: $execution_time seconds"




    echo "------Refinement and filtering (10x Parallel) --------"
    start_time=$SECONDS

    # Function to check for stuck items
    check_and_retry_stuck_items() {
        echo "Checking for stuck refinement items..."
        
        # Find lock files older than 10 minutes and remove them
        find ${HDOCK_DIR}_iter_$i -name ".*.lock" -mmin +10 -delete 2>/dev/null || true
        
        # Run a single instance to pick up any remaining work
        echo "Running cleanup instance to process any remaining refinement items..."
        python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_refinement_after_docking.py \
            --hdock_models ${HDOCK_DIR}_iter_$i \
            --iteration $i \
            --inference_summary ${SUMMARY_FILE_INFERENCE} \
            --dataset_json ${DATASET} \
            --top_n $n
    }

    # Clean up any leftover lock files first
    find ${HDOCK_DIR}_iter_$i -name ".*.lock" -delete 2>/dev/null || true

    # Run 10 instances in parallel - each will find unique work
    for instance in {1..5}; do
        echo "Starting refinement instance $instance"
        python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_refinement_after_docking.py \
            --hdock_models ${HDOCK_DIR}_iter_$i \
            --iteration $i \
            --inference_summary ${SUMMARY_FILE_INFERENCE} \
            --dataset_json ${DATASET} \
            --top_n $n &
        sleep 3 # Just to stagger startup
    done

    wait
    echo "All 10 instances completed!"

    # Check for stuck items and retry
    check_and_retry_stuck_items

    # Clean up lock files
    find ${HDOCK_DIR}_iter_$i -name ".*.lock" -delete 2>/dev/null || true

    elapsed=$((SECONDS - start_time))
    echo "Refinement completed in $elapsed seconds"


    echo "-----------Inference-----------"
    start_time=$SECONDS

    # Create a directory for each fold in the results directory
    mkdir -p ${RESULT_DIR}_iteration_$i

    if [ "$MODEL" = "DiffAb" ]; then
    
        conda activate diffab

        # conda run -n diffab python 
        python ${DYMEAN_CODE_DIR}/models/pipeline/cdr_models/NanoDesigner_diffab_inference_original.py \
            --dataset ${DATASET} \
            --config ${CONFIG} \
            --out_dir ${RESULT_DIR}_iteration_$i \
            --hdock_models ${HDOCK_DIR}_iter_$i \
            --diffab_code_dir ${DIFFAB_CODE_DIR}  \
            --dymean_code_dir ${DYMEAN_CODE_DIR}  \
            --summary_dir ${SUMMARY_FILE_INFERENCE}  \
            --gpu 0  \
            --iteration $i 

        conda deactivate

    elif [ "$MODEL" = "ADesigner" ]; then


        conda activate dymean2

        cd $ADESIGNER_CODE_DIR
        python ${ADESIGNER_CODE_DIR}/generate_pipeline.py \
            --ckpt ${CKPT} \
            --test_set ${DATASET} \
            --out_dir ${RESULT_DIR}_iteration_$i \
            --hdock_models ${HDOCK_DIR}_iter_$i \
            --rabd_topk 5  \
            --mode "1*1"  \
            --rabd_sample 20 \
            --config ${CONFIG} \
            --iteration $i 


    else
        echo "Error: Unknown model type '${MODEL}'. Please choose from DiffAb or ADesigner. Revise config file"
        exit 1
    fi

    wait
    end_time=$SECONDS
    execution_time=$((end_time - start_time))
    echo "Inference took approximately: $execution_time seconds"

    cd ${DYMEAN_CODE_DIR}
    
    echo --------- Conducting Side chain Packing ----------------

    #Run script
    start_time=$SECONDS
    python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_side_chain_packing.py \
        --summary_json ${SUMMARY_FILE_INFERENCE} \
        --out_file ${SUMMARY_FILE_PACKED} \
        --test_set ${DATASET} \
        --cdr_model ${MODEL}
    wait

    end_time=$SECONDS
    execution_time=$((end_time - start_time))
    echo "Side chain Packing after Side Chain Packing took approximately: $execution_time seconds"





    echo "------Side Chain Packing/Refinement --------"
    start_time=$SECONDS


    # Function to check for stuck items
    check_and_retry_stuck_items() {
        echo "Checking for stuck items..."
        # Find lock files older than 10 minutes and remove them
        find $(dirname "$OUT_FILE")/.locks -name "*.lock" -mmin +10 -delete 2>/dev/null || true
        # Run a single instance to pick up any remaining work
        echo "Running cleanup instance to process any remaining items..."
        python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_refinement_after_side_chain_packing_parallel.py \
            --in_file ${SUMMARY_FILE_PACKED} \
            --out_file ${SUMMARY_FILE_PACKED_REFINED} \
            --cdr_model ${MODEL}
    }

    # Clean up any leftover lock files first
    find $(dirname "$OUT_FILE") -name ".*.lock" -delete 2>/dev/null || true
    find $(dirname "$OUT_FILE")/.locks -name "*.lock" -delete 2>/dev/null || true

    # Run 10 instances in parallel
    for instance in {1..5}; do
        echo "Starting side chain packing instance $instance"
        python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_refinement_after_side_chain_packing_parallel.py \
            --in_file ${SUMMARY_FILE_PACKED} \
            --out_file ${SUMMARY_FILE_PACKED_REFINED} \
            --cdr_model ${MODEL} &
        sleep 3
    done

    wait
    echo "All 10 instances completed!"

    # Check for stuck items and retry
    check_and_retry_stuck_items

    # Final cleanup
    find $(dirname "$OUT_FILE") -name ".*.lock" -delete 2>/dev/null || true
    find $(dirname "$OUT_FILE")/.locks -name "*.lock" -delete 2>/dev/null || true

    elapsed=$((SECONDS - start_time))
    echo "Side chain packing completed in $elapsed seconds"



    echo "----------Iter Evaluation-----------"

    start_time=$SECONDS

    mkdir -p ${DATA_DIR}${DATASET_NAME}/csv_iter_$i

    python ${DYMEAN_CODE_DIR}/NanoDesigner_metrics_computations.py \
        --test_set ${DATASET} \
        --summary_json ${SUMMARY_FILE_PACKED_REFINED} \
        --hdock_models ${HDOCK_DIR}_iter_$i \
        --cdr_type ${CDRS} \
        --iteration $i \
        --cdr_model ${MODEL} \
        --csv_dir ${DATA_DIR}${DATASET_NAME}/csv_iter_$i 

    
    wait
    execution_time=$((end_time - start_time))
    end_time=$SECONDS
    echo "Evaluation took approximately: $execution_time seconds"

    echo "----------Best Mutants Selection-----------"
    python ${DYMEAN_CODE_DIR}/models/pipeline/NanoDesigner_best_mutant_selection.py \
        --dataset_json ${DATASET} \
        --top_n $N \
        --hdock_models ${HDOCK_DIR}_iter_$i \
        --iteration $i \
        --csv_dir ${DATA_DIR}/csv_iter_$i \
        --objective ${MAX_OBJECTIVE} 

    wait  


done

