# NanoDesigner: Resolving the Complex-CDR Interdependency with Iterative Refinement

![Alt text](https://github.com/Melissaurious/NanoDesigner/blob/main/NanoDesigner.png)

NanoDesigner is an end-to-end workflow for the design and optimization of nanobodies. It integrates key stages—Structure Prediction, Docking, CDR Generation,and Side-Chain Packing—into an iterative framework based on an Expectation Maximization algorithm. Our method effectively tackles an often overlooked interdependency
challenge where accurate docking presupposes a priori knowledge of the CDR conformation, while effective CDR generation relies on accurate docking outputs to guide its design.


## Table of Contents
- [Installation](#installation)
- [External Tool Installation](#external-tool-installation)
- [Data Download and Preprocess](#data-download-and-preprocess)
- [Inference-tool assessment](#inference)
- [NanoDesigner](#nanodesigner)
- [Citation](#citation)
- [License](#license)
- [Credits](#credits)



## Installation


```bash
git clone https://github.com/Melissaurious/NanoDesigner.git
cd NanoDesigner
```

### Create and activate the Conda environment for graph-based models
```bash
conda env create -f nanodesigner_1.yml -n nanodesigner1
conda activate nanodesigner1
```

### Create and activate the Conda environment for difussion-based model
```bash
conda env create -f nanodesigner_2.yml -n nanodesigner2
conda activate nanodesigner2
```

## External Tool Installation

The following repositories and software tools are required for NanoDesigner. Clone them into the `NanoDesigner` directory and follow the installation steps provided in their respective webpages:

- [IgFold](https://github.com/Graylab/IgFold) - *Trained models already included in nanodesigner1 conda environment.*
- [DockQ](https://github.com/bjornwallner/DockQ)
- [Rosetta](https://docs.rosettacommons.org/demos/latest/tutorials/install_build/install_build)
- [FoldX](https://foldxsuite.crg.eu/products#foldx)
- [HDOCK](http://huanglab.phys.hust.edu.cn/software/hdocklite/)
- [dr_sasa_n](https://github.com/nioroso-x3/dr_sasa_n) - *Follow the instructions in the repository to compile this tool.*

After installing the tools, ensure to update the `dyMEAN/configs.py` file with the full paths to the installed tools.

Source code for TMscore evatuation is at `dyMEAN/evaluation/`, please compile as:
```bash
g++ -static -O3 -ffast-math -lm -o evaluation/TMscore evaluation/TMscore.cpp
```


## Data Download and Preprocess

The data download and preparation steps are necessary to replicate our data processing, filtering, and preparation for training. All required instructions are included in the following scripts.

### 1. Preprocess the Data
Use the `download_and_process_data_reduced.py` script to download SAbDab structures, extract CDR information, and analyze antibody-antigen interactions.
The preprocessing pipeline performs structure validation, CDR extraction, interaction analysis, and epitope mapping. It automatically handles data download, quality filtering, and generates the final dataset with mapped epitopes from CDR-antigen contacts.

#### Usage
```bash
python download_and_process_data_reduced.py \
    --output_folder <output_directory> \
    --type <Antibody|Nanobody> \
    --numbering <imgt|chothia> \
```

#### Parameters
* `--output_folder`: Directory for all output files (required)
* `--type`: Structure type to analyze - `Antibody` or `Nanobody` (default: `Nanobody`)
* `--numbering`: Numbering scheme - `imgt` or `chothia` (default: `imgt`)
* `--max_resolution`: Maximum resolution threshold in Å (default: `4.0`)
* `--tsv_file`: Custom SAbDab TSV file (auto-downloaded if not provided)
* `--raw_structures_dir`: Custom PDB structures directory (auto-downloaded if not provided)

#### Output
The main result is `CDRH3_interacting_[type]_[numbering]_unique.json` containing the final filtered dataset with CDR sequences, interaction data, and epitope mappings (input for the next step).


### 2. Split the Data
After downloading and preprocessing the data, use the split_data.py script to create train/validation/test splits with sequence-based clustering.
The splitting script performs sequence similarity clustering on either CDRH3 or antigen sequences, then creates k-fold cross-validation splits ensuring that similar sequences are kept within the same split. This prevents data leakage where the model might see similar sequences during training and testing.
#### Usage
```bash
python ./Data_download_and_processing/scripts/split_data.py \
    --data_files <path_to_json_file(s)> \
    --immuno_molecule <Antibody|Nanobody|Antibody Nanobody> \
    --cluster_targets <CDRH3|Ag> \
    --out_dir <output_directory> \
```



## Training and Inference

We provide trained models for evaluating different training configurations (please unzip all_checkpoints.tar.gz inside the NanoDesigner directory):

**Training Data Variants:**
- Nanobodies-only
- Nanobodies + Antibodies
- Antibodies fine-tuned on Nanobodies

**Clustering Thresholds:**
- Antigen (Ag) sequence clustering (95%, 80%, 60%)
- CDRH3 sequence clustering (40%, 30%, 20%)

These configurations systematically assess model performance across:
1) Different training data compositions
2) Varying levels of structural diversity (CDRH3 loops)
3) Different antigenic similarity levels

### Running Assessments
To generate evaluation results:
1. Use `NanoDesigner_2/scripts/CDRH3_model_assessment.bash`
2. Specify the required variables:
   - Working directory
   - Tool selection
   - Sequence clustering type (Ag or CDRH3)
   - Threshold value

### Results Analysis
The performance metrics presented in:
- Supplementary Material Tables 2 & 3
- Main Text Table 1

were generated using `NanoDesigner_2/Tool_assesment_results/generate_tables_experiment_1.ipynb`. This notebook processes the assessment outputs to produce the comparative tables in our publication.





## NanoDesigner

NanoDesigner is an end-to-end workflow designed for both **de novo** and **optimization** cases in nanobody-antigen complex design. The workflow script is located in the `scripts` folder and can be executed as follows:

```bash
bash scripts/NanoDesigner.sh your_working_directory/denovo_epitope_info/7eow_8pwh_example/7eow_8pwh_ep_1.json
```


### Test Cases

The workflow requires a script and a JSON file containing the necessary information for each entry (a nanobody-antigen complex or nanobody scaffold and antigen structure). 

- **De Novo Design**: In cases where the 3D structure of a nanobody-antigen complex is absent (referred to as "de novo" design), the input JSON file can be generated using the notebook `jupyter_notebooks/prepare_NanoDesigner_inputs_Denovo.ipynb`. This notebook guides you through creating a properly formatted JSON file.

- **Optimization Cases**: For existing complexes, simply select a relevant line from the dataset-generated JSON files (prepared during the [Data Download and Preprocess](#data-download-and-preprocess) stage) and use it to create an input JSON file.

All required information for both cases should be obtained during the data download and preprocessing stage. Ensure the configuration files (`config_files`) are updated as needed to reflect your setup.

For proof of concepts of NanoDesigner, please download and employ DiffAb or ADesigner trained models found [here](https://drive.google.com/drive/folders/1kGK3rV138lG8vQpGAtHv5oNP_a11Gr01?usp=share_link).


We highly encourage to keep a constant number of total number of designs across iterations for simplicity:

```python
R = 50  # Number of randomized nanobodies (Initialization step)
N = 15  # Top best mutants to proceed with to subsequent iterations
d = 100 # Docked models to generate with Hdock
n = 5   # Top docked models to feed to inference stage
k_iteration_1 = 3   # Number of predictions obtained from CDR Generation stage at iteration 1
k_iteration_x = 10  # Number of predictions obtained from CDR Generation stage at iteration x

Rxnxk = 750 (Iteration 1)
Nxnxk = 750 (Iteration X)
```

### NanoDesigner test cases:
*De novo design escenario:
CDRH3 or 3CDRs design with ΔG optimization objective.

<div style="text-align: center;">
  <img src="https://github.com/Melissaurious/NanoDesigner/blob/main/nanodesigner_test_cases_2025.png" alt="Alt Text" width="800">
</div>



## Citation
TODO

## License
TODO

## Credits

This codebase is primarily based on the following deep learning tools. We thank the authors for their contributions:

- [Diffab](https://github.com/luost26/diffab)  
- [dyMEAN](https://github.com/THUNLP-MT/dyMEAN)  
- [ADesigner](https://github.com/A4Bio/ADesigner) 

We also acknowledge the rest of tools and software that played a crucial role in the workflow employed in this study.
We sincerely thank the authors of these tools for their invaluable work, which made this project possible.




