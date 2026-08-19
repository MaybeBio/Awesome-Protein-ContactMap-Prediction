# Docker Deployment

> **Relevant source files**
> * [Dockerfile](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile)
> * [docs/docker_installation.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1)
> * [docs/infer_json_format.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1)
> * [requirements.txt](https://github.com/bytedance/Protenix/blob/c3bfc365/requirements.txt)

This document provides comprehensive guidance for deploying Protenix using Docker containers. It covers building Docker images from the provided `Dockerfile`, configuring GPU support, managing volume mounts, and running inference workloads in containerized environments.

## Docker Image Architecture

The Protenix Docker image is built using a multi-layered approach that optimizes for both build efficiency and runtime performance. The base image includes PyTorch 2.7.1 and CUDA 12.6.3 support, with additional layers for system dependencies, Python packages, and specialized components like CUTLASS.

### Image Component Hierarchy

```mermaid
flowchart TD

BASE["pytorch:2.7.1-cu12.6.3-py3.11-ubuntu22.04"]
SYSPACK["System Packages"]
GCC["g++, gcc, libc6-dev"]
MAKE["make"]
BIO_TOOLS["hmmer, kalign"]
TORCH["torch==2.7.1"]
TORCHVISION["torchvision==0.22.1"]
TORCHAUDIO["torchaudio==2.7.1"]
CUEQUIV["cuequivariance-ops-torch-cu12==0.8.0"]
CUEQUIV_TORCH["cuequivariance-torch==0.8.0"]
SCIPY["scipy>=1.9.0"]
NUMPY["numpy==2.4.1"]
PANDAS["pandas==2.3.1"]
SKLEARN["scikit-learn==1.7.1"]
BIOPYTHON["biopython==1.85"]
BIOTITE["biotite==1.4.0"]
ESM["fair-esm==2.0.0"]
RDKIT["rdkit==2025.9.3"]
GEMMI["gemmi==0.6.7"]
CUTLASS["/opt/cutlass (v3.5.1)"]
CUTLASS_ENV["CUTLASS_PATH=/opt/cutlass"]

BASE --> SYSPACK
BIO_TOOLS --> TORCH
TORCHVISION --> CUEQUIV
CUEQUIV_TORCH --> SCIPY
SKLEARN --> BIOPYTHON
GEMMI --> CUTLASS

subgraph subGraph6 ["External Tools Layer"]
    CUTLASS
    CUTLASS_ENV
    CUTLASS --> CUTLASS_ENV
end

subgraph subGraph5 ["Bioinformatics Layer"]
    BIOPYTHON
    BIOTITE
    ESM
    RDKIT
    GEMMI
    BIOPYTHON --> BIOTITE
    BIOTITE --> ESM
    ESM --> RDKIT
    RDKIT --> GEMMI
end

subgraph subGraph4 ["Scientific Computing Layer"]
    SCIPY
    NUMPY
    PANDAS
    SKLEARN
    SCIPY --> NUMPY
    NUMPY --> PANDAS
    PANDAS --> SKLEARN
end

subgraph subGraph3 ["Specialized Dependencies Layer"]
    CUEQUIV
    CUEQUIV_TORCH
    CUEQUIV --> CUEQUIV_TORCH
end

subgraph subGraph2 ["PyTorch Layer"]
    TORCH
    TORCHVISION
    TORCHAUDIO
    TORCH --> TORCHVISION
    TORCH --> TORCHAUDIO
end

subgraph subGraph1 ["System Dependencies Layer"]
    SYSPACK
    GCC
    MAKE
    BIO_TOOLS
    SYSPACK --> GCC
    SYSPACK --> MAKE
    SYSPACK --> BIO_TOOLS
end

subgraph subGraph0 ["Base Layer"]
    BASE
end
```

**Sources:** [Dockerfile L1-L33](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile#L1-L33)

 [requirements.txt L1-L33](https://github.com/bytedance/Protenix/blob/c3bfc365/requirements.txt#L1-L33)

## Container Runtime Environment

The Docker container establishes a complete runtime environment with optimized Python settings, timezone configuration, and essential paths for the Protenix system components.

### Environment and Volume Mapping

```mermaid
flowchart TD

DEBIAN["DEBIAN_FRONTEND=noninteractive"]
TZ["TZ=Asia/Shanghai"]
PYTHON_OPT["PYTHONDONTWRITEBYTECODE=1"]
PYTHON_BUF["PYTHONUNBUFFERED=1"]
CUTLASS_PATH["CUTLASS_PATH=/opt/cutlass"]
WORKSPACE["/app"]
OPT_CUTLASS["/opt/cutlass"]
DEV_SHM["/dev/shm"]
PROTENIX_CLI["protenix"]
HOST_CODE["$(pwd)"]
HOST_SHM["/dev/shm"]

DEBIAN --> WORKSPACE
TZ --> WORKSPACE
PYTHON_OPT --> WORKSPACE
PYTHON_BUF --> WORKSPACE
CUTLASS_PATH --> OPT_CUTLASS
HOST_CODE --> WORKSPACE
HOST_SHM --> DEV_SHM
WORKSPACE --> PROTENIX_CLI

subgraph subGraph3 ["Volume Mounts"]
    HOST_CODE
    HOST_SHM
end

subgraph subGraph2 ["Entry Points"]
    PROTENIX_CLI
end

subgraph subGraph1 ["File System Layout"]
    WORKSPACE
    OPT_CUTLASS
    DEV_SHM
end

subgraph subGraph0 ["Environment Variables"]
    DEBIAN
    TZ
    PYTHON_OPT
    PYTHON_BUF
    CUTLASS_PATH
end
```

**Sources:** [Dockerfile L3-L8](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile#L3-L8)

 [Dockerfile L25](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile#L25-L25)

 [docs/docker_installation.md L27-L32](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L27-L32)

## Building and Pulling the Image

The Docker image can be built from the provided `Dockerfile` or pulled from the official registry.

### Building from Source

The `Dockerfile` uses a pre-configured PyTorch image and installs necessary system tools like `hmmer` and `kalign` for MSA generation.

```
docker build -t protenix:local .
```

### Using Pre-built Image

The recommended approach for quick deployment:

```
docker pull ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4
```

### Dependency Installation Phases

The `Dockerfile` installs dependencies in a specific order to leverage layer caching:

| Phase | Components | Purpose |
| --- | --- | --- |
| **System** | `g++`, `gcc`, `libc6-dev`, `make`, `postgresql`, `hmmer`, `kalign` | Compilation tools and bioinformatics search utilities |
| **Python** | `requirements.txt` | Core libraries including `torch`, `cuequivariance`, and `rdkit` |
| **Specialized** | `CUTLASS v3.5.1` | Cloned to `/opt/cutlass` for optimized GPU kernel support |

**Sources:** [Dockerfile L11-L22](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile#L11-L22)

 [Dockerfile L29-L33](https://github.com/bytedance/Protenix/blob/c3bfc365/Dockerfile#L29-L33)

 [docs/docker_installation.md L13-L17](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L13-L17)

## Running the Container

Container execution requires proper GPU configuration and volume mounts to ensure Protenix can access input JSON data and write structural outputs.

### Basic Container Execution

Run the container with all GPUs enabled and the current Protenix directory mounted:

```
docker run --gpus all -it \    -v "$(pwd)":/app \    -v /dev/shm:/dev/shm \    ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4 \    /bin/bash
```

### GPU Configuration Requirements

1. **NVIDIA Container Toolkit**: Must be installed on the host to enable `--gpus all`.
2. **CUDA Compatibility**: The base image uses CUDA 12.6.3. Ensure host drivers are compatible.
3. **Verification**: Run `nvidia-smi` inside the container to verify visibility.

**Sources:** [docs/docker_installation.md L5-L11](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L5-L11)

 [docs/docker_installation.md L27-L33](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L27-L33)

## Protenix Installation in Container

Since the Docker image does not include the Protenix source code by default, it must be installed in editable mode after mounting the repository.

### Installation Process

Once inside the container:

```
cd /apppip install -e .
```

This setup:

* Installs the `protenix` command line tool.
* Links the local source code to the environment.
* Allows immediate verification via `protenix --help`.

**Sources:** [docs/docker_installation.md L35-L43](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L35-L43)

## Container Workflow

The typical workflow for using Protenix in a Docker container follows a sequence from initialization to inference.

```mermaid
sequenceDiagram
  participant User
  participant Docker
  participant Container
  participant ProtenixCLI
  participant GPU

  User->>Docker: "docker run --gpus all -it"
  Docker->>Container: "Start container with GPU & /app mount"
  User->>Container: "pip install -e ."
  Container->>ProtenixCLI: "Register CLI entry point"
  User->>ProtenixCLI: "protenix prep input.json"
  ProtenixCLI->>Container: "Run MSA (hmmer/kalign) & update JSON"
  User->>ProtenixCLI: "protenix pred input.json"
  ProtenixCLI->>GPU: "Execute diffusion sampling"
  GPU->>ProtenixCLI: "Return coordinates & confidence"
  ProtenixCLI->>Container: "Write .cif and .json to /app/output"
  Container->>User: "Prediction complete"
```

**Sources:** [docs/docker_installation.md L27-L43](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L27-L43)

 [docs/infer_json_format.md L143-L147](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L143-L147)

## Troubleshooting

### Shared Memory (/dev/shm)

Protenix and its underlying libraries (like PyTorch and certain bioinformatics tools) may require significant shared memory for multi-processing. Always use `-v /dev/shm:/dev/shm` or `--shm-size` to prevent "Bus error" or "No space left on device" errors during MSA generation or data loading.

### Pathing and Absolute Paths

When providing MSA paths in the input JSON (e.g., `pairedMsaPath`), it is highly recommended to use **absolute paths** relative to the container's file system (e.g., starting with `/app/`).

**Sources:** [docs/docker_installation.md L29-L30](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/docker_installation.md?plain=1#L29-L30)

 [docs/infer_json_format.md L66-L68](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/infer_json_format.md?plain=1#L66-L68)