# Establishment and Evaluation of an End-to-End Computational Workflow for the Design and Optimization of Nanobodies: NanoDesigner

![Alt text](https://github.com/Melissaurious/NanoDesigner/blob/main/NanoDesigner_.png)

## Table of Contents
- [Installation](#installation)
- [Data Download and Preprocessing](#data-download-and-preprocessing)
- [Training and Inference](#training-and-inference)
- [Workflow Description](#workflow-description)
- [Tools Compilation](#tools-compilation)
- [Examples](#examples)
- [Citation](#citation)
- [License](#license)



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

The following repositories and software tools are required for NanoDesigner. Clone them into the `NanoDesigner` directory and follow the installation steps provided in their respective repositories:

- [IgFold](https://github.com/Graylab/IgFold)
- [DockQ](https://github.com/bjornwallner/DockQ)
- [Rosetta](https://docs.rosettacommons.org/demos/latest/tutorials/install_build/install_build)
- [FoldX](https://foldxsuite.crg.eu/products#foldx)
- [HDOCK](http://huanglab.phys.hust.edu.cn/software/hdocklite/)
- [dr_sasa_n](https://github.com/nioroso-x3/dr_sasa_n) - *Follow the instructions in the repository to compile this tool.*

After installing the tools, ensure to update the `dyMEAN/configs.py` file with the full paths to the installed tools.


## Data Download and Preprocess

The data download and preparation steps are necessary to replicate our data processing, filtering, and preparation for training. All required instructions are included in the provided Jupyter notebooks.

### 1. Preprocess the Data
- Open the notebook located at `jupyter_notebooks/process_datasets.ipynb`.
- Follow the instructions in the notebook to download and preprocess the datasets.

### 2. Split the Data
- Once preprocessing is complete, open the notebook at `jupyter_notebooks/split_data.ipynb`.
- Use this notebook to split the processed data into training and testing sets.

### Notes:
- Ensure [Jupyter Notebook](https://jupyter.org/install) is installed. To check, run:
  ```bash
  jupyter --version


## Training and Inference

The `scripts` folder contains `.sh` scripts for both training and inference workflows used in the study. These scripts are configured for each of the tools employed in this study and are designed to facilitate a 10-fold cross-validation setup.

### 1. Training
- **Location**: Training scripts for each tool are located in the `scripts` directory.
- **Configuration**: Update the file paths and any necessary parameters inside the scripts. This includes specifying paths for datasets, output directories and additional variables.
- **10-Fold Cross-Validation**: The scripts are pre-configured to implement a 10-fold cross-validation strategy. Refer to *-Data Download and Preprocess*.

### 2. Inference
- **Location**: Inference scripts for each tool are also available in the `scripts` directory.
- **Configuration**: Make sure that the paths across the training and inference scripts match. The folder specified in the training script dictates the location of the generated checkpoints, which will be used during inference.
- **Manual Checkpoint Selection**: For GNN-based tools, selection of the best checkpoint must be done manually. Refer to the instructions provided in the script files for guidance.



To run the training or inference for a specific tool, execute the corresponding script as in the example:
```bash
bash scripts/train_tool.sh


