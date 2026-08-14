---
slug:2-quick-start
blog_type:normal
---


AlphaFold 3 是 Google DeepMind 在结构预测领域的最新突破，能够预测蛋白质、RNA、DNA 和配体的结构及其相互作用。本指南将帮助您在几分钟内完成首次 AlphaFold 3 预测。

## 前置条件

在开始之前，请确保您具备以下条件：

- **硬件**：一台配备 NVIDIA GPU（计算能力 8.0+，例如 A100、H100）、64GB+ RAM 和约 1TB 存储空间的 Linux 机器
- **软件**：Docker 或 Singularity
- **访问权限**：从 Google 获取模型参数（您需要单独请求这些参数）

参考资料：[README.md](README.md)，[docs/installation.md](docs/installation.md)

## 五步完成首次预测

### 1. 克隆仓库

```bash
git clone https://github.com/google-deepmind/alphafold3.git
cd alphafold3
```

### 2. 下载遗传数据库

AlphaFold 3 需要多个遗传数据库进行搜索。仓库中包含一个脚本，用于下载所有数据库：

```bash
./fetch_databases.sh $HOME/public_databases
```

<CgxTip>此步骤将下载约 252GB 数据并解压至约 630GB。考虑到下载时间约为 45 分钟（在良好网络速度下），建议在 screen/tmux 会话中运行。</CgxTip>

参考资料：[docs/installation.md](docs/installation.md)，[fetch_databases.sh](fetch_databases.sh)

### 3. 构建 Docker 容器

```bash
docker build -t alphafold3 -f docker/Dockerfile .
```

### 4. 创建简单的输入文件

为您的输入和输出创建目录：

```bash
mkdir -p $HOME/af_input $HOME/af_output
```

创建一个名为 `$HOME/af_input/fold_input.json` 的文件，内容如下：

```json
{
  "name": "我的首次预测",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
      }
    }
  ],
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1
}
```

此示例预测一个同源二聚体（同一蛋白质链的两个副本）。

参考资料：[README.md](README.md)，[docs/input.md](docs/input.md)

### 5. 运行 AlphaFold 3

```bash
docker run -it \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume $HOME/public_databases:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/fold_input.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

将 `<MODEL_PARAMETERS_DIR>` 替换为您存储 AlphaFold 3 模型参数的路径。

参考资料：[README.md](README.md)，[run_alphafold.py](run_alphafold.py)

## 理解输出结果

运行成功后，您将在输出目录中找到一个名为 `my_first_prediction` 的目录，其中包含：

- `my_first_prediction_model.cif`：最高排名的预测结构，以 mmCIF 格式保存（可在 PyMOL、ChimeraX 或 VMD 等分子查看器中查看）
- `my_first_prediction_confidences.json`：详细的置信度指标
- `my_first_prediction_summary_confidences.json`：摘要置信度指标
- 每个种子和样本的子目录，包含额外的预测结果

关键置信度指标包括：

- **pLDDT**（0-100）：每个原子的置信度评分，越高越好
- **pTM**（0-1）：整个结构的预测 TM 评分，>0.5 表示折叠正确
- **ipTM**（0-1）：预测的界面 TM 评分，>0.8 表示对界面有高置信度

参考资料：[docs/output.md](docs/output.md)

## 灵活选项

### 仅运行数据管道或仅运行推理

AlphaFold 3 的过程有两个主要阶段，可以分别运行：

```bash
# 仅运行数据管道（MSA 搜索，模板搜索 - CPU 密集型）
python run_alphafold.py --run_data_pipeline=true --run_inference=false ...

# 仅运行推理（需要数据管道结果 - GPU 密集型）
python run_alphafold.py --run_data_pipeline=false --run_inference=true ...
```

这种分离允许您在一台机器上运行 CPU 密集型搜索，在另一台机器上运行 GPU 密集型预测。

### 处理不同类型的分子

AlphaFold 3 可以预测以下结构的结构：

- **蛋白质**：使用 `"protein": {"id": "A", "sequence": "..."}` 指定
- **RNA**：使用 `"rna": {"id": "B", "sequence": "..."}` 指定
- **DNA**：使用 `"dna": {"id": "C", "sequence": "..."}` 指定
- **配体**：使用 CCD 码或 SMILES 字符串指定

### 探索高级功能

对于更复杂的预测，您可以指定：

- 具有不同序列的多条链
- 使用 CCD 码或 SMILES 字符串指定的配体
- 翻译后修饰
- 自定义 MSA 和模板
- 实体间的共价键

参考资料：[docs/input.md](docs/input.md)

## 故障排除技巧

- **"Failed to construct RDKit reference structure"**：尝试增加构象迭代次数，使用 `--conformer_max_iterations=1000`
- **MSA 工具错误**：确保数据库具有正确的权限（`chmod 755 -R public_databases`）
- **内存错误**：对于大型蛋白质/复合物，考虑使用更多 RAM 的机器

参考资料：[docs/known_issues.md](docs/known_issues.md)

## 下一步

- **探索输入选项**：查看 [输入文档](docs/input.md) 了解更复杂的示例
- **分析置信度指标**：学习如何解读 [输出数据](docs/output.md) 以评估预测质量
- **运行多个种子**：在 `modelSeeds` 中设置多个值以生成多样化的预测

AlphaFold 3 通过准确预测复杂生物分子结构，为结构生物学开辟了新的可能性。使用这一强大工具推进您在蛋白质-蛋白质、蛋白质-RNA/DNA 和蛋白质-配体相互作用领域的研究。