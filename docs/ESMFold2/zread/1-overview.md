---
slug:1-overview
blog_type:normal
---


ESM 是 EvolutionaryScale 的开源蛋白质建模框架，围绕两个互补的模型家族构建：**ESM3**，一个跨序列、结构和功能进行联合推理的多模态生成模型；以及 **ESM C**，一个表征模型，可创建丰富的生物嵌入，且效率较前代有显著提升。`esm` Python 包 (v3.2.4) 提供了统一的 SDK，支持本地推理和基于云的 API 访问，实现了从单 GPU 快速原型验证到 Forge 或 AWS SageMaker 大规模批处理的全场景覆盖。无论你是要从部分提示词设计新型蛋白质，还是为下游预测任务提取嵌入，本仓库都为你提供了相应的工具和抽象，让你能通过一致且类型化的接口来完成这些操作。

来源: [README.md](/README.md#L1-L50), [esm/__init__.py](/esm/__init__.py#L1-L2), [pyproject.toml](/pyproject.toml#L1-L10)

## 两大模型家族，一个统一框架

ESM 生态系统围绕两个截然不同但又可互操作的模型家族构建，每个家族针对不同的核心用例而设计。**ESM3** 是一个生成式掩码语言模型：你可以在任意组合的轨道（序列、结构、二级结构、SASA、功能）上提供部分信息，模型会对掩码位置进行迭代采样，直到所有轨道补全。**ESM C** 是一个表征模型：它将蛋白质序列编码为高质量的嵌入，可用于属性预测、变异效应分析或相似性搜索等下游任务。这两个模型家族共享相同的 `ESMProtein` 数据模型和 SDK 抽象，因此在它们之间切换或组合其输出时，只需极少的代码修改。

来源: [esm/models/esm3.py](/esm/models/esm3.py#L118-L125), [esm/models/esmc.py](/esm/models/esmc.py#L29-L50)

### ESM3：多模态生成模型

ESM3 作为在离散 token 序列上运行的**生成式掩码语言模型**。它在六个输入轨道——序列、结构 token、二级结构 (SS8)、溶剂可及表面积 (SASA)、功能注释 (InterPro) 和残基注释——上进行联合推理，并且可以通过部分填充这些轨道的任意子集作为提示词。该模型的 Transformer 骨干网络在其初始层使用几何注意力来对 3D 结构进行条件化，从而实现对原子坐标的 SE(3) 不变推理。在最大规模下，ESM3 使用 1.07×10²⁴ FLOPs 在 27.8 亿个蛋白质上进行了训练，并且开源的 14 亿参数变体 (`esm3-sm-open-v1`) 已可用于本地推理。

![ESM3 图解](_assets/esm3_diagram.png)

来源: [README.md](/README.md#L69-L86), [esm/models/esm3.py](/esm/models/esm3.py#L44-L97)

### ESM C：表征模型

ESM C (Cambrian) 是一个**序列到嵌入**的模型，旨在作为 ESM2 的直接替代品，但具有显著的效率提升。3 亿参数的 ESM C 提供了与 6.5 亿参数 ESM2 相当的性能，而 6 亿参数的变体则可与 30 亿参数的 ESM2 相媲美。ESM C 使用标准的 Transformer 堆叠（没有几何注意力层），使其更轻量、更快速。它暴露了 `encode()` → `logits()` → `embeddings` 作为其核心工作流，并支持 Flash Attention 以在 GPU 上进一步加速。ESM C 提供开源权重变体（3 亿、6 亿参数），并可通过 Forge API 或 SageMaker 使用更大的 60 亿参数模型。

来源: [README.md](/README.md#L147-L155), [esm/models/esmc.py](/esm/models/esmc.py#L29-L65)

## 模型对比

| 特性 | ESM3 | ESM C |
|---------|------|-------|
| **核心用途** | 可控的蛋白质生成 | 蛋白质表征学习 |
| **输入模态** | 序列 + 结构 + SS + SASA + 功能 + 残基 | 仅序列 |
| **输出轨道** | 全部 6 个轨道 + 嵌入 | 序列 logits + 嵌入 + 隐藏状态 |
| **几何注意力** | 是（初始层） | 否 |
| **生成** | 迭代掩码采样 | 不支持 |
| **开源权重** | 14 亿 (`esm3-sm-open-v1`) | 3 亿, 6 亿 |
| **Forge API 规格** | 14 亿, 70 亿, 980 亿 | 60 亿 |
| **SageMaker** | — | 3 亿, 6 亿, 60 亿 |
| **Flash Attention** | — | 是（自动检测） |
| **核心 API** | `ESM3InferenceClient` | `ESMCInferenceClient` |

| ESM3 模型 | 参数量 | d_model | 注意力头数 | 层数 | 发布时间 |
|------------|-----------|---------|-------|--------|---------|
| esm3-large-2024-03 | 98B | — | — | — | 2024-03 |
| esm3-medium-2024-08 | 7B | — | — | — | 2024-08 |
| esm3-small-2024-08 (open) | 1.4B | 1536 | 24 | 48 | 2024-08 |

| ESM C 模型 | 参数量 | d_model | 注意力头数 | 层数 | 发布时间 |
|-------------|-----------|---------|-------|--------|---------|
| esmc-6b-2024-12 | 6B | — | — | 80 | 2024-12 |
| esmc-600m-2024-12 | 600M | 1152 | 18 | 36 | 2024-12 |
| esmc-300m-2024-12 | 300M | 960 | 15 | 30 | 2024-12 |

来源: [README.md](/README.md#L52-L67), [esm/pretrained.py](/esm/pretrained.py#L77-L117), [esm/models/esm3.py](/esm/models/esm3.py#L118-L138), [esm/models/esmc.py](/esm/models/esmc.py#L29-L50)

## 高层架构

下图展示了 ESM 的主要组件，以及数据如何从原始蛋白质输入流经分词、Transformer 核心、输出解码，最终还原为内容丰富的 `ESMProtein` 对象。阴影区域代表模型的内部前向传播；其外部的一切都属于 SDK 层，负责处理编码、生成编排和解码。

```mermaid
graph TB
    subgraph SDK["SDK 层
        direction TB
        ESMProtein["ESMProtein<br/>原始数据：序列，坐标，<br/>SS，SASA，功能注释"]
        Forge["Forge API 客户端<br/>(esm/sdk/forge.py)"]
        Local["本地推理<br/>(ESM3 / ESMC 模型)"]
        SM["SageMaker 客户端<br/>(esm/sdk/sagemaker.py)"]
    end

    subgraph Tokenization["分词
        SeqTok["序列分词器"]
        StructTok["结构分词器<br/>+ VQ-VAE 编码器"]
        SSTok["SS8 分词器"]
        SASATok["SASA 分词器"]
        FuncTok["功能分词器<br/>(InterPro)"]
        ResTok["残基注释<br/>分词器"]
    end

    subgraph Core["Transformer 核心
        GeomAttn["几何注意力<br/>(SE(3) 不变)"]
        StdAttn["标准多头<br/>注意力 + SwiGLU FFN"]
        OutHeads["输出头<br/>(6 个回归头)"]
    end

    subgraph Decode["解码
        StructDec["VQ-VAE 结构<br/>解码器"]
        FuncDec["功能 token<br/>解码器"]
    end

    ESMProtein -->|"encode()"| Tokenization
    Tokenization -->|"ESMProteinTensor"| Core
    Core -->|"ESMOutput / ESMCOutput"| OutHeads
    OutHeads -->|"decode()"| Decode
    Decode -->|"ESMProtein"| ESMProtein

    Forge -.->|"HTTP API"| Tokenization
    Local -.->|"直接调用"| Tokenization
    SM -.->|"AWS 端点"| Tokenization
```

<CgxTip>`encode()` → `forward()` → `decode()` 管道是基础模式。当你调用 `model.generate()` 时，它会通过迭代掩码采样多次编排此管道，逐步取消 token 的掩码，直到目标轨道补全。</CgxTip>

来源: [esm/sdk/api.py](/esm/sdk/api.py#L200-L260), [esm/models/esm3.py](/esm/models/esm3.py#L200-L350), [esm/tokenization/__init__.py](/esm/tokenization/__init__.py#L1-L67), [esm/layers/transformer_stack.py](/esm/layers/transformer_stack.py#L1-L50)

## 项目结构

本仓库由四个主要子系统组成：**模型实现**、**SDK 和部署客户端**、**分词管道**以及**工具层**。教程和实用示例位于顶层的 `cookbook/` 目录中。

```
esm/
├── models/              # 模型架构
│   ├── esm3.py          # ESM3 多模态生成模型
│   ├── esmc.py          # ESM C 表征模型
│   ├── function_decoder.py   # InterPro 功能 token 解码器
│   └── vqvae.py         # VQ-VAE 结构编码器/解码器
│
├── sdk/                 # 用于推理的客户端 SDK
│   ├── api.py           # ESMProtein, ESMProteinTensor, 客户端抽象基类
│   ├── forge.py         # Forge API 远程客户端
│   ├── sagemaker.py     # AWS SageMaker 部署客户端
│   └── __init__.py      # esm.sdk.client() 工厂函数
│
├── tokenization/        # 输入/输出分词器
│   ├── sequence_tokenizer.py
│   ├── structure_tokenizer.py
│   ├── ss_tokenizer.py
│   ├── sasa_tokenizer.py
│   ├── function_tokenizer.py
│   └── residue_tokenizer.py
│
├── layers/              # 神经网络构建模块
│   ├── attention.py     # 标准多头注意力
│   ├── geom_attention.py # SE(3) 几何注意力
│   ├── blocks.py        # UnifiedTransformerBlock
│   ├── transformer_stack.py  # Transformer 堆叠
│   ├── rotary.py        # 旋转位置嵌入
│   └── ffn.py           # SwiGLU 前馈网络
│
├── utils/               # 编码、解码、采样、常量
├── pretrained.py        # 模型注册表和 from_pretrained 加载器
└── widgets/             # Jupyter notebook 可视化工具

cookbook/
├── tutorials/           # 逐步 notebook 教程
├── snippets/            # ESM3、ESM C、折叠的最简代码示例
└── local/               # 带有原始前向传播的本地推理示例
```

来源: [esm/pretrained.py](/esm/pretrained.py#L123-L135), [esm/tokenization/__init__.py](/esm/tokenization/__init__.py#L1-L67), [esm/sdk/__init__.py](/esm/sdk/__init__.py#L1-L37)

## 部署选项

`esm` SDK 提供了三种部署路径，它们都共享相同的 `ESM3InferenceClient` 或 `ESMCInferenceClient` 接口。这意味着你可以仅修改一行代码就将本地模型替换为 Forge 客户端——无需更改任何其他代码。

| 部署方式 | 实例化方法 | 最适用场景 |
|------------|-------------------|----------|
| **本地** | `ESM3.from_pretrained("esm3-sm-open-v1").to("cuda")` | 快速原型验证、完整模型访问、离线使用 |
| **Forge API** | `esm.sdk.client("esm3-medium-2024-08", token=...)` | 最大模型（980 亿、60 亿参数）、无需 GPU、批量执行 |
| **SageMaker** | `ESM3SageMakerClient(endpoint_name=..., model=...)` | 商业用途、专用 GPU、AWS 集成 |

<CgxTip>Forge 和 SageMaker 客户端均支持通过 `async_` 前缀（如 `async_generate`、`async_encode`）调用异步方法来处理并发工作负载。请使用 `batch_executor()` 上下文管理器来执行高吞吐量的 Forge 任务。</CgxTip>

来源: [esm/sdk/__init__.py](/esm/sdk/__init__.py#L1-L37), [esm/sdk/forge.py](/esm/sdk/forge.py#L1-L50), [README.md](/README.md#L104-L130)

## 阅读指南

本手册分为三个部分。请从**入门**开始设置你的环境并了解核心数据模型，然后进入**深度剖析**了解架构细节，或者跳转至**动态**获取最新更新和社区信息。

**入门**（推荐阅读顺序）：
1. [概述](1-overview) — 你正在这里
2. [快速开始](2-quick-start) — 安装、配置并运行你的首次推理
3. [ESMProtein 数据模型](3-esmprotein-data-model) — 了解 `ESMProtein` / `ESMProteinTensor` 类型

**深度剖析**（根据你的兴趣探索）：
- **模型内部机制**：[架构概述](7-architecture-overview) → [ESM3 多模态生成模型](8-esm3-multimodal-generative-model) 或 [ESM C 表征模型](9-esm-c-representation-model)
- **分词**：[序列与结构分词器](10-sequence-and-structure-tokenizers) → [VQ-VAE 结构编码](11-vq-vae-structure-encoding) → [功能与残基注释 Token](12-function-and-residue-annotation-tokens)
- **Transformer 核心**：[几何注意力与 SE(3) 不变性](13-geometric-attention-and-se-3-invariance) → [Transformer 堆叠设计](14-transformer-stack-design) → [旋转嵌入与 SwiGLU](15-rotary-embeddings-and-swiglu)
- **生成**：[迭代掩码采样](16-iterative-masked-sampling) → [噪声调度与脱掩策略](17-noise-schedules-and-unmasking-strategies) → [生成配置参考](18-generation-configuration-reference)
- **SDK 与部署**：[Forge API 客户端](19-forge-api-client) → [本地推理客户端](20-local-inference-client) → [SageMaker 与批量执行](21-sagemaker-and-batch-execution)