# Installation

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

### 1) Prerequisites

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

##### GPU users

Build the image:

```bash
docker build -f Dockerfile.gpu -t bridge:gpu .
```

Launch a container with GPU support:

```bash
docker run --rm -it --gpus all bridge:gpu
```

##### CPU users

Build the image:

```bash
docker build -f Dockerfile.cpu -t bridge:cpu .
```

Launch a container with CPU support:

```bash
docker run --rm -it bridge:cpu
```

### Sanity-Check for Environment Setup

To verify that the environment has been set up correctly and avoid dependency conflicts, especially with PyTorch and PyTorch Geometric, you can check the installed versions directly in the command line. Run the following commands to ensure that the necessary libraries are correctly installed and compatible.

**Run these commands:**

```bash
# Check PyTorch version and CUDA availability
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())"

# Check PyTorch Geometric version
python -c "import torch_geometric; print('torch-geometric:', torch_geometric.__version__)"
```

This will display the installed versions of PyTorch and PyTorch Geometric, as well as the CUDA version and availability. An example output might look like this:

```bash
torch: 2.0.1+cu117 cuda: 11.7 cuda_available: True
torch-geometric: 2.6.1
```

If the versions match the recommended ones in the prerequisites section, the PyTorch and PyTorch Geometric are correctly set up.