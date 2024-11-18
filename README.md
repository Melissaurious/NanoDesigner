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

To prepare the datasets required for NanoDesigner, follow these steps:

### 1. Preprocess the Data
- Open the notebook located at `jupyter_notebooks/process_datasets.ipynb`.
- Follow the instructions in the notebook to download and preprocess the required datasets.

### 2. Split the Data
- Once the data preprocessing is complete, open the notebook located at `jupyter_notebooks/split_data.ipynb`.
- Use this notebook to split the data into training and testing sets.

### Notes:
- Make sure you have [Jupyter Notebook](https://jupyter.org/install) installed. If you're unsure, you can check by running:
  ```bash
  jupyter --version

