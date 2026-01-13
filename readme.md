<h1 align="center">
BRIDGE: Bridging Sequence–Structure Motifs and Genetic Variants for Genome-wide Dynamic RNA–Protein Interaction Profiling
</h1>

<p align="center">

<!-- GitHub Actions CI -->

<a href="https://github.com/wangyb97/BRIDGE/actions/workflows/ci.yml">
    <img src="https://github.com/wangyb97/BRIDGE/actions/workflows/ci.yml/badge.svg">
  </a>

<!-- Language -->

<a href="https://github.com/wangyb97/BRIDGE">
    <img src="https://img.shields.io/badge/BRIDGE-Python-blue.svg">
  </a>

<!-- Stars -->

<a href="https://github.com/wangyb97/BRIDGE/stargazers">
    <img src="https://img.shields.io/github/stars/wangyb97/BRIDGE?style=flat&color=yellow">
  </a>

<!-- Forks -->

<a href="https://github.com/wangyb97/BRIDGE/network/members">
    <img src="https://img.shields.io/github/forks/wangyb97/BRIDGE?style=flat&color=orange">
  </a>

<!-- Issues -->

<a href="https://github.com/wangyb97/BRIDGE/issues">
    <img src="https://img.shields.io/github/issues/wangyb97/BRIDGE?style=flat&color=informational">
  </a>

<!-- License -->

<a href="https://github.com/wangyb97/BRIDGE/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/wangyb97/BRIDGE?color=green">
  </a>

<!-- Contributors -->

<a href="https://github.com/wangyb97/BRIDGE#contributors-">
    <img src="https://img.shields.io/badge/All_Contributors-1-purple.svg">
  </a>

</p>

## 📑 Table of Contents

- [Overview](#overview)
- [Environment Setup](#%EF%B8%8Fenvironment-setup)
  - [Tested Environment](#tested-environment)
  - [Prerequisites](#1-prerequisites)
  - [Recommended installation (Conda)](#2-recommended-installation-conda)
  - [Running in docker (Optional)](#3-running-in-docker-optional)
- [Data &amp; Resources](#data--resources)
- [Usage](#usage)
  - [Train](#1-train)
  - [Validate](#2-validate-evaluate-a-saved-model)
  - [Dynamic Transfer Prediction](#3-dynamic-transfer-prediction-crosscell-type)
  - [Variant-Aware Scoring](#4-variant-aware-scoring)
  - [Motif construction](#5-motif-construction)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🔬Overview

<p align="center">
  <img src="framework.png" alt="BRIDGE framework" width="80%">
</p>

BRIDGE is an advanced multimodal deep learning framework for predicting dynamic RNA–protein binding landscapes and assessing the functional impact of genetic variants across multiple human cell types. It leverages a unified architecture that integrates:

- **Sequence embeddings** from pretrained Transformer models to capture rich contextual nucleotide representations.
- **RNA secondary structure features** to model the spatial and thermodynamic constraints on RBP binding.
- **Motif priors** derived from *de novo* motif discovery (STREME) to incorporate known binding patterns.
- **Biochemical profiles** capturing experimental signals such as reactivity, accessibility, and conservation.
- **Graph-based attention modeling** to represent long-range dependencies between nucleotides via token-wise relational graphs.

By fusing these complementary modalities, BRIDGE can accurately characterize both conserved and dynamic binding preferences, enabling:

- **End-to-end model training and evaluation** on large-scale eCLIP/HITS-CLIP datasets.
- **Dynamic cross-cell-type transfer prediction**, where the model generalizes to unseen cellular contexts without fine-tuning.
- **Variant-aware inference**, assessing the functional impact of genetic variants (e.g., SNVs) on RBP binding to facilitate disease and trait association studies.
- **Explicit motif extraction** highlighting dynamic sequence–structure patterns learned from the fused modalities.

This multimodal and interpretable design positions BRIDGE as a powerful tool for dissecting post-transcriptional regulation, guiding functional genomics studies, and prioritizing disease-associated variants with potential regulatory impact.

## ⚙️Environment Setup

### Tested Environment

BRIDGE is platform-agnostic and can run on Linux, macOS, and Windows (via WSL). Below are the hardware and software we have tested to ensure reproducibility:

| GPU                     | VRAM  | Driver version | CUDA version |
| ----------------------- | ----- | -------------- | ------------ |
| NVIDIA A40              | 48 GB | 550.54.14      | 12.4         |
| NVIDIA L40              | 48 GB | 550.54.14      | 12.4         |
| Quadro RTX 6000         | 24 GB | 550.54.14      | 12.4         |
| NVIDIA GeForce RTX 3090 | 24 GB | 580.95.05      | 13.0         |
| NVIDIA TITAN RTX        | 24 GB | 580.95.05      | 13.0         |

#### 1) Prerequisites

The following table summarizes the key software dependencies and the tested versions for BRIDGE:

| Package         | Stable version |
| --------------- | -------------- |
| python          | 3.10.10        |
| torch           | 2.0.1          |
| torchvision     | 0.15.2         |
| torch-geometric | 2.6.1          |
| transformers    | 4.41.2         |
| tokenizers      | 0.19.1         |
| numpy           | 1.23.5         |
| scipy           | 1.10.1         |
| pandas          | 2.0.0          |
| scikit-learn    | 1.6.1          |
| biopython       | 1.85           |
| viennarna       | 2.6.4          |
| tqdm            | 4.67.1         |
| matplotlib      | 3.4.1          |
| seaborn         | 0.13.2         |
| captum          | 0.7.0          |
| shap            | 0.41.0         |

### 2) Recommended installation (Conda)

```bash
# Create and activate an environment
conda env create -f BRIDGE.yml
conda activate BRIDGE
```

### 3) Running in docker (Optional)

If you prefer a fully containerized environment, BRIDGE can also run in Docker.

#### Step 1: Install Docker

Download and install the latest Docker version for your platform:
[Docker Installers](https://docs.docker.com/get-started/get-docker/).

To enable GPU access inside Docker, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

#### Step 2: Build and run the Docker image

Build the image:

```bash
docker build -t bridge:latest .
```

Launch a container with GPU support:

```bash
docker run --rm -it --gpus all bridge:latest
```

## 📂Data & Resources

To ensure reproducibility and ease of use, we provide all necessary resources pre-packaged for BRIDGE:

| Resource         | Path (after extraction)              | Description                                   |
| ---------------- | ------------------------------------ | --------------------------------------------- |
| 🔹 Raw Data      | `BRIDGE/dataset/`                  | Input RNA tracks for model training/testing   |
| 🔹 Variant FASTA | `BRIDGE/dataset_variant/`          | Variant-aware inference inputs (FASTA format) |
| 🔹 RBPformer     | `BRIDGE/RBPformer/`                | Pretrained Transformer model                  |
| 🔹 Motif Priors  | `BRIDGE/utils/motif_prior/output/` | Precomputed motif prior files for all RBPs    |
| 🔹 Model Files   | `BRIDGE/results/model/`            | Trained BRIDGE models                         |

📥 **Download from**:

- 🔗 [figshare DOI (v2)](https://doi.org/10.6084/m9.figshare.29819843.v2)
- 🌐 [BRIDGE online portal](http://rbp.aibio-lab.com/app/api/download/index/)

After downloading, extract the files to the corresponding locations under BRIDGE root directory.

## 🚀Usage

Below are GPU and CPU versions of each command. Use --use_cpu to force CPU even when GPUs are available.

### 1) Train

#### GPU version (default)

```bash
python main.py \
    --train \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --device_num 0 \
    --early_stopping 20 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --lr 0.001
```

#### CPU version (force CPU)

```bash
python main.py \
    --train \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --use_cpu \
    --early_stopping 20 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --lr 0.001
```

**Flags explained**

- `--train`
  Run end-to-end training.
- `--data_path ./dataset`
  Folder containing `<DATA_FILE>_pos.fa` and `<DATA_FILE>_neg.fa`.
- `--data_file AUH_HepG2`
  Dataset stem. Loader looks for `AUH_HepG2_pos.fa` and `AUH_HepG2_neg.fa` in `--data_path`.
- `--use_cpu`
  Force CPU mode even if CUDA is available.
- `--device_num 0`
  Index of the CUDA device (equivalent to `torch.cuda.set_device(0)`).
- `--early_stopping 20`
  Patience for early stopping (stop if validation metric does not improve for 20 epochs).
- `--Transformer_path ./RBPformer`
  Local directory of the pretrained Transformer/“RBPformer” checkpoint (tokenizer + model).
- `--model_save_path ./results/model`
  Directory to save best models.
- `--lr 0.001`
  Initial learning rate.

**Outputs**

- Model checkpoints saved under `--model_save_path`.
- Training logs/metrics.

### 2) Validate (Evaluate a saved model)

#### GPU version (default)

```bash
python main.py \
    --validate \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --device_num 0 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
```

#### CPU version (force CPU)

```bash
python main.py \
    --validate \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --use_cpu \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
```

**Flags explained**

- `--validate`
  Run validation only. The code will load the model from `--model_save_path` (ensure a trained model exists).
- Other flags are identical in meaning to the Train section.

### 3) Dynamic Transfer Prediction (Cross–cell-type)

#### GPU version (default)

```bash
python main.py \
    --dynamic_predict \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --device_num 0 \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
```

#### CPU version (force CPU)

```bash
python main.py \
    --dynamic_predict \
    --data_path ./dataset \
    --data_file AUH_HepG2 \
    --use_cpu \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model
```

**Flags explained**

- `--dynamic_predict`
  Perform zero-shot / cross–cell-type inference.
- `--data_file AUH_HepG2`
  Target dataset stem for dynamic prediction.
- Remaining flags carry the same meaning as above.

**Notes**

- Ensure the correct source-trained model is available under `--model_save_path`.

### 4) Variant-Aware Scoring

#### GPU version

```bash
python variant_aware.py \
    --after_variation \
    --fasta_sequence_path ./dataset_variant/AUH_HepG2.fa \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --variant_out_file ./results/variants/AUH_HepG2_after_mut.txt \
    --device cuda:3
```

#### CPU version (force CPU)

```bash
python variant_aware.py \
    --after_variation \
    --fasta_sequence_path ./dataset_variant/AUH_HepG2.fa \
    --Transformer_path ./RBPformer \
    --model_save_path ./results/model \
    --variant_out_file ./results/variants/AUH_HepG2_after_mut.txt \
    --device cpu
```

**Flags explained**

- `--after_variation`
  Use the post-mutation sequences to compute variant-aware scores.
- `--fasta_sequence_path`
  Path to the FASTA-like file to score.
- `--Transformer_path` / `--model_save_path`
  Same as above; specify the Transformer directory and the trained BRIDGE model location.
- `--variant_out_file`
  Output path for scored variants.
- `--device`
  Control whether to use CPU or GPU mode.

### 5) Motif construction

#### Step 1: Generate attention maps for motif visualization

```bash
export KMER=1
export MODEL_PATH=../RBPformer
export DATA_PATH=examples/AARS_K562
export PREDICTION_PATH=examples/AARS_K562
cd motif_construction/

python run_finetune.py \
    --model_type dna \
    --tokenizer_name $MODEL_PATH \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_visualize \
    --visualize_data_dir $DATA_PATH \
    --visualize_models $KMER \
    --data_dir $DATA_PATH \
    --max_seq_length 101 \
    --per_gpu_pred_batch_size=4 \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 96
```

**Flags explained**

- `--model_type dna`
  Sets the model type for processing DNA/RNA sequences.
- `--tokenizer_name` / `--model_name_or_path`
  Paths to the pretrained RBPformer tokenizer and model.
- `--task_name dnaprom`
  Task identifier (inherited from training scripts; here for promoter-like sequence processing).
- `--do_visualize`
  Enables generation of attention maps for visualization.
- `--visualize_data_dir`
  Input data folder containing sequences.
- `--visualize_models`
  K-mer length (here set via `$KMER`).
- `--data_dir`
  Same as input data path.
- `--max_seq_length`
  Maximum sequence length.
- `--per_gpu_pred_batch_size`
  Prediction batch size per GPU.
- `--output_dir`
  Directory for saving outputs.
- `--predict_dir`
  Directory for prediction results.
- `--n_process`
  Number of parallel processes for data handling.

#### Step 2: Discover motifs from attention maps

```bash
export KMER=1
echo $(pwd)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export DATA_PATH=examples/AARS_K562
export ATTENTION_PATH=examples/AARS_K562
export OUTPUT_PATH=examples/AARS_K562/result

python motif/Discovery_motifs.py \
    --input_rna_dir $DATA_PATH \
    --attention_dir $ATTENTION_PATH \
    --motif_length 6 \
    --min_region_len 6 \
    --fdr_cutoff 0.01 \
    --min_motif_count 3 \
    --align_all_ties \
    --output_motif_dir $OUTPUT_PATH \
    --verbose
```

**Flags explained**

- `--input_rna_dir`
  Directory containing RNA sequences used for motif extraction.
- `--attention_dir`
  Directory with saved attention scores from Step 1.
- `--motif_length`
  Target motif length.
- `--min_region_len`
  Minimum contiguous region length for candidate motifs.
- `--fdr_cutoff`
  False discovery rate threshold for motif selection.
- `--min_motif_count`
  Minimum number of motif occurrences to be considered significant.
- `--align_all_ties`
  Align motifs even if multiple sequences have identical scores.
- `--output_motif_dir`
  Output folder for discovered motifs.
- `--verbose`
  Enables detailed logging.

## 📜License

This project is licensed under the MIT License.

## 📚Citation

If you use BRIDGE in your research, please cite the accompanying manuscript.

```
Unpublished yet
```

## 🤝Acknowledgements

- Hugging Face Transformers for tokenizer/model loading.
- PyTorch & PyTorch Geometric for deep learning and GNN components.

**Email**: yubo23@mails.jlu.edu.cn; lixt314@jlu.edu.cn.
