---
slug:14-docker-deployment
blog_type:normal
---


AlphaFold 3 提供了一个基于 Docker 的全面部署解决方案，简化了安装过程，并确保在不同环境中的执行一致性。本指南将引导您使用 Docker 部署 AlphaFold 3，从初始设置到运行您的第一个结构预测。

## 前置条件

在使用 Docker 部署 AlphaFold 3 之前，请确保您的系统满足以下要求：

- Linux 操作系统（AlphaFold 3 不支持其他操作系统）
- 具有计算能力 8.0 或更高的 NVIDIA GPU（推荐 A100 或 H100）
- 至少 64GB RAM（更长的序列需要更多内存）
- 最高 1TB 的磁盘空间用于遗传数据库（推荐使用 SSD 存储）
- 主机机器上安装了 CUDA 12.6
- 支持 NVIDIA 容器的 Docker

**硬件推荐**：AlphaFold 3 已验证可在单个 NVIDIA A100 80GB 或 H100 80GB GPU 上处理最多 5,120 个标记的输入。遗传搜索阶段可能会消耗大量内存，因此拥有足够的内存对于最佳性能至关重要。

## 设置支持 NVIDIA 的 Docker

### 1. 安装 Docker

对于 Ubuntu 22.04 LTS 或更高版本：

```sh
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

测试您的 Docker 安装：

```sh
sudo docker run hello-world
```

### 2. 启用无根 Docker（推荐）

```sh
sudo apt-get install -y uidmap systemd-container

sudo machinectl shell $(whoami)@ /bin/bash -c 'dockerd-rootless-setuptool.sh install && sudo loginctl enable-linger $(whoami) && DOCKER_HOST=unix:///run/user/1001/docker.sock docker context use rootless'
```

### 3. 安装 NVIDIA 驱动和 Docker 支持

安装 NVIDIA 驱动：

```sh
sudo apt-get -y install alsa-utils ubuntu-drivers-common
sudo ubuntu-drivers install

sudo nvidia-smi --gpu-reset
```

使用 `nvidia-smi` 验证安装。如果遇到问题，您可能需要重启系统。

安装 NVIDIA 容器工具包：

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
```

验证 Docker 中的 GPU 访问：

```sh
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 获取 AlphaFold 3 资源

### 1. 克隆仓库

```sh
git clone https://github.com/google-deepmind/alphafold3.git
```

### 2. 下载遗传数据库

AlphaFold 3 需要多个遗传数据库以实现最佳性能。使用提供的脚本：

```sh
cd alphafold3  # 导航到仓库目录
./fetch_databases.sh [<DB_DIR>]
```

如果未指定，默认 `<DB_DIR>` 为 `$HOME/public_databases`。

**重要**：数据库目录不应是 AlphaFold 3 仓库目录的子目录。总下载大小约为 252 GB，解压后大小约为 630 GB。请确保您有足够的磁盘空间、带宽和权限。

为了提高性能，考虑将数据库移动到 SSD：

```sh
src/scripts/gcp_mount_ssd.sh [<SSD_MOUNT_PATH>]
src/scripts/copy_to_ssd.sh [<DB_DIR>] [<SSD_DB_DIR>]
```

### 3. 获取模型参数

要访问 AlphaFold 3 模型参数：
1. 完成 [请求表单](https://forms.gle/svvpY4u2jsHEwWYS6)
2. 等待批准（通常需要 2-3 个工作日）
3. 将参数下载到您选择的目录（`<MODEL_PARAMETERS_DIR>`）

与数据库一样，这个目录不应位于 AlphaFold 3 仓库目录内。

## 构建和运行 Docker 容器

### 1. 构建 Docker 镜像

在 AlphaFold 3 仓库目录中：

```sh
docker build -t alphafold3 -f docker/Dockerfile .
```

这将构建一个包含所有必需的 Python 依赖项、HMMER（带有自定义序列限制补丁）和其他必要组件的容器。

### 2. 准备输入文件

根据 [输入文档](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) 创建一个输入 JSON 文件，并将其放置在一个目录中（例如，`$HOME/af_input`）。

### 3. 运行 AlphaFold 3

```sh
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <DB_DIR>:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

其中：
- `$HOME/af_input`：包含输入 JSON 文件的目录
- `$HOME/af_output`：输出结果的目录
- `<MODEL_PARAMETERS_DIR>`：包含下载的模型参数的目录
- `<DB_DIR>`：包含遗传数据库的目录

您可能需要创建输出目录并设置适当的权限：

```sh
mkdir -p $HOME/af_output
chmod 755 $HOME/af_input $HOME/af_output
```

## 优化性能

### 使用 SSD 存储数据库

为了提高遗传搜索性能，将数据库放在 SSD 上。如果您有一些数据库在 SSD 上，而其他数据库在较慢的磁盘上：

```sh
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume <SSD_DB_DIR>:/root/public_databases \
    --volume <DB_DIR>:/root/public_databases_fallback \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --db_dir=/root/public_databases \
    --db_dir=/root/public_databases_fallback \
    --output_dir=/root/af_output
```

此配置允许 AlphaFold 3 从 SSD 访问数据库，并在需要时回退到更大、更慢的磁盘。

### 高级配置选项

要查看所有可用的命令行选项：

```sh
docker run alphafold3 python run_alphafold.py --help
```

## 替代方案：使用 Singularity 而非 Docker

对于不推荐或不可用 Docker 的环境（如某些 HPC 系统），您可以使用 Singularity：

### 1. 安装 Singularity

```sh
wget https://github.com/sylabs/singularity/releases/download/v4.2.1/singularity-ce_4.2.1-jammy_amd64.deb
sudo dpkg --install singularity-ce_4.2.1-jammy_amd64.deb
sudo apt-get install -f
```

### 2. 从 Docker 镜像构建 Singularity 容器

```sh
docker run -d -p 5000:5000 --restart=always --name registry registry:2
docker tag alphafold3 localhost:5000/alphafold3
docker push localhost:5000/alphafold3

SINGULARITY_NOHTTPS=1 singularity build alphafold3.sif docker://localhost:5000/alphafold3:latest
```

验证 GPU 访问：

```sh
singularity exec --nv alphafold3.sif sh -c 'nvidia-smi'
```

### 3. 使用 Singularity 运行 AlphaFold 3

```sh
singularity exec \
     --nv \
     --bind $HOME/af_input:/root/af_input \
     --bind $HOME/af_output:/root/af_output \
     --bind <MODEL_PARAMETERS_DIR>:/root/models \
     --bind <DB_DIR>:/root/public_databases \
     alphafold3.sif \
     python run_alphafold.py \
     --json_path=/root/af_input/fold_input.json \
     --model_dir=/root/models \
     --db_dir=/root/public_databases \
     --output_dir=/root/af_output
```

## 故障排除

- **挂载源路径错误**：如果看到类似“Error response from daemon: error while creating mount source path”的错误，请确保指定的目录存在并具有适当的权限。
- **GPU 访问问题**：确保 NVIDIA 驱动正确安装，并且容器具有 GPU 访问权限。在主机和容器内都运行 `nvidia-smi` 以验证。
- **数据库权限**：如果遇到 MSA 工具的错误，请确保您的数据库目录具有完全的读写权限（例如，`sudo chmod 755 --recursive <DB_DIR>`）。

## 摘要

使用 Docker 部署 AlphaFold 3 提供了一个一致、可复现的环境，用于蛋白质结构预测。按照本指南，您可以设置必要的依赖项，构建 Docker 容器，并运行您自己的数据预测。为了获得最佳性能，考虑使用 SSD 存储数据库，并确保您的 GPU 和 RAM 符合推荐规格。