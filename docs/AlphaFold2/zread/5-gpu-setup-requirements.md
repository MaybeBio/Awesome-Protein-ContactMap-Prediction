---
slug:5-gpu-setup-requirements
blog_type:normal
---


本指南概述了高效运行 AlphaFold 所需的 GPU 硬件和软件要求。正确的 GPU 配置对于在蛋白质结构预测任务中实现最佳性能至关重要。

## GPU 硬件要求

AlphaFold 需要配备充足显存的现代 NVIDIA GPU 来处理蛋白质结构预测工作负载。GPU 显存容量直接影响可处理的蛋白质最大尺寸。

| GPU 显存 | 推荐用例 | 最大蛋白质长度 |
|----------|----------|----------------|
| 8GB+ | 小型蛋白质（< 350 个残基） | 约 350 个残基 |
| 16GB+ | 中型蛋白质（350-700 个残基） | 约 700 个残基 |
| 24GB+ | 大型蛋白质（700-1400 个残基） | 约 1400 个残基 |
| 32GB+ | 超大型蛋白质（> 1400 个残基） | 约 2000+ 个残基 |

<CgxTip>对于支持高达 640 个残基的大型复合物的 AlphaFold-Multimer v2.3.0，建议使用至少 16GB 显存的 GPU 以获得最佳性能。</CgxTip>

## 软件依赖

### CUDA 和 cuDNN 要求

AlphaFold Docker 容器使用特定 CUDA 版本构建以确保最佳兼容性：

```dockerfile
ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
```

**必需组件：**
- CUDA 12.2.2 或兼容版本
- cuDNN 8 运行时
- NVIDIA Container Toolkit（用于 Docker GPU 支持）

### 安装步骤

1. **安装 NVIDIA Container Toolkit**：
   ```bash
   # 请参考适用于你的 Linux 发行版的官方 NVIDIA 文档
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **验证 GPU 检测**：
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **为非 root 用户配置 Docker**：
   ```bash
   # 设置非 root 用户使用 Docker
   sudo usermod -aG docker $USER
   ```

## GPU 配置选项

### Docker 运行时参数

AlphaFold 通过 Docker 启动脚本 [run_docker.py](docker/run_docker.py#L31) 提供灵活的 GPU 配置：

```python
flags.DEFINE_bool('use_gpu', True, '启用 NVIDIA 运行时以使用 GPU 运行。')
flags.DEFINE_bool('enable_gpu_relax', True, '如果启用 GPU，则在 GPU 上运行 relax。')
flags.DEFINE_string('gpu_devices', 'all', '传递给 NVIDIA_VISIBLE_DEVICES 的设备列表，以逗号分隔。')
```

**GPU 使用场景：**

- **完全 GPU 加速**：`--use_gpu=True --enable_gpu_relax=True`
- **仅 CPU 松弛**：`--use_gpu=True --enable_gpu_relax=False`
- **指定 GPU 设备**：`--gpu_devices="0,1"`（使用 GPU 0 和 1）
- **多 GPU 设置**：配置 `--gpu_devices` 指定可用 GPU

### 内存优化

Docker 容器 [Dockerfile](docker/Dockerfile#L58) 中的 JAX 库配置包含 GPU 特定优化：

```dockerfile
pip3 install --upgrade --no-cache-dir \
  jax==0.4.26 \
  jaxlib==0.4.26+cuda12.cudnn89 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

<CgxTip>对于内存受限环境，考虑使用 `reduced_dbs` 预设搭配较小的基因数据库以减少内存占用。</CgxTip>

## 性能注意事项

### GPU 内存管理

AlphaFold 的内存使用量随以下因素扩展：
- 蛋白质序列长度
- MSA 序列数量
- 模板数量
- 模型复杂度（单体 vs 多聚体）

### Amber 松弛 GPU 支持

Amber 松弛过程 [relax.py](alphafold/relax/relax.py#L35) 支持 GPU 加速：

```python
def __init__(self, *, use_gpu: bool):
    # GPU 加速的结构优化
```

基于 GPU 的松弛可显著提升速度，但需要额外 GPU 显存。如果内存有限，可通过 `--enable_gpu_relax=False` 禁用。

## GPU 问题排查

### 常见问题与解决方案

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| NVIDIA Container Toolkit | `docker: Error response from daemon: could not select device driver` | 重新安装 NVIDIA Container Toolkit 并重启 Docker 守护进程 |
| GPU 显存不足 | 预测过程中出现内存不足错误 | 使用较小蛋白质、减少批次大小或禁用 GPU 松弛 |
| CUDA 版本不匹配 | 驱动程序/库版本不匹配 | 验证主机与容器间的 CUDA 兼容性 |

### 验证命令

```bash
# 检查 Docker 中的 GPU 可用性
docker run --rm --gpus all nvidia/cuda:12.2.2-base nvidia-smi

# 测试 AlphaFold 容器 GPU 访问
docker build -f docker/Dockerfile -t alphafold .
docker run --rm --gpus all alphafold python -c "import jax; print(jax.devices())"
```

## 后续步骤

配置完 GPU 设置后，请继续 [数据库配置](4-database-configuration) 下载必需的基因数据库，如需回顾容器设置可返回 [Docker 安装](3-docker-installation)。

要全面了解 AlphaFold 的计算要求，请查阅 [技术说明](docs/technical_note_v2.3.0)，其中详细介绍了不同硬件配置的性能特征和优化策略。