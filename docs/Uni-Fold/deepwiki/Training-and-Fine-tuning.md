# Training and Fine-tuning

> **Relevant source files**
> * [finetune_monomer.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_monomer.sh)
> * [finetune_multimer.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_multimer.sh)
> * [train_monomer.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer.sh)
> * [train_monomer_demo.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer_demo.sh)
> * [train_multimer.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer.sh)
> * [train_multimer_demo.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer_demo.sh)
> * [unifold/config.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py)

This document covers how to train Uni-Fold models from scratch and fine-tune existing models. It explains the configuration system, training scripts, model variants, and the underlying training infrastructure. For information about parameter conversion between different frameworks, see [Parameter Conversion](/dptech-corp/Uni-Fold/6.3-parameter-conversion). For details about the specific model architectures being trained, see [Model Architecture](/dptech-corp/Uni-Fold/5-model-architecture).

## Overview

Uni-Fold provides a comprehensive training system built on top of the `unicore-train` framework that supports both training from scratch and fine-tuning existing models. The system handles multiple model variants including monomer and multimer models, with configurable architectures and training parameters.

## Training Configuration System

The training system is controlled by a hierarchical configuration system defined in [unifold/config.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py)

 The configuration separates concerns into data processing, model architecture, and loss function parameters.

### Configuration Architecture

```mermaid
flowchart TD

A["base_config"]
B["model_config"]
C["Model Variants"]
D["model_1"]
E["model_2"]
F["multimer"]
G["multimer_ft"]
H["model_1_ft"]
I["data"]
J["common"]
K["train"]
L["eval"]
M["predict"]
N["model"]
O["input_embedder"]
P["evoformer_stack"]
Q["structure_module"]
R["loss"]
S["fape"]
T["distogram"]
U["plddt"]
V["d_pair=128"]
W["d_msa=256"]
X["d_single=384"]
Y["max_recycling_iters=3"]

B --> I
B --> N
B --> R
B --> V

subgraph subGraph2 ["Global Parameters"]
    V
    W
    X
    Y
end

subgraph subGraph1 ["Configuration Sections"]
    I
    J
    K
    L
    M
    N
    O
    P
    Q
    R
    S
    T
    U
    I --> J
    I --> K
    I --> L
    I --> M
    N --> O
    N --> P
    N --> Q
    R --> S
    R --> T
    R --> U
end

subgraph subGraph0 ["Configuration Hierarchy"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    C --> G
    C --> H
end
```

Sources: [unifold/config.py L26-L466](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L26-L466)

### Model Variants

The configuration system supports multiple model variants, each with specific parameter settings:

| Variant | Purpose | Key Differences |
| --- | --- | --- |
| `model_1` | Base monomer model | Standard AlphaFold architecture |
| `model_1_ft` | Fine-tuned monomer | Larger MSA (5120), larger crop (384) |
| `model_2` | Alternative monomer | Different feature processing |
| `multimer` | Protein complexes | Chain-aware processing, PAE head |
| `multimer_ft` | Fine-tuned multimer | Optimized for complex prediction |

Sources: [unifold/config.py L480-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L480-L672)

### Training vs Inference Configuration

The configuration system differentiates between training and inference modes:

```mermaid
flowchart TD

A["data.train"]
B["crop=True"]
C["crop_size=256"]
D["supervised=True"]
E["block_delete_msa=True"]
F["data.predict"]
G["crop=False"]
H["supervised=False"]
I["block_delete_msa=False"]
J["num_ensembles=2"]
K["data.eval"]
L["crop=False"]
M["supervised=True"]
N["num_ensembles=1"]

subgraph subGraph0 ["Data Processing Modes"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    J
    K
    L
    M
    N
    A --> B
    A --> C
    A --> D
    A --> E
    F --> G
    F --> H
    F --> I
    F --> J
    K --> L
    K --> M
    K --> N
end
```

Sources: [unifold/config.py L208-L226](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L208-L226)

 [unifold/config.py L176-L189](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L176-L189)

 [unifold/config.py L191-L207](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L191-L207)

## Training Scripts

### Training from Scratch

Uni-Fold provides separate scripts for training monomer and multimer models:

#### Monomer Training

```mermaid
flowchart TD

A["train_monomer.sh"]
B["unicore-train"]
C["--task af2"]
D["--loss af2"]
E["--arch af2"]
F["--model-name {variant}"]
G["Environment Variables"]
H["n_gpu"]
I["MASTER_PORT"]
J["update_freq"]
K["total_step=80000"]
L["lr=1e-3"]
M["Training Parameters"]
N["--optimizer adam"]
O["--lr-scheduler exponential_decay"]
P["--bf16"]
Q["--ema-decay 0.999"]

A --> B
B --> C
B --> D
B --> E
B --> F
G --> H
G --> I
G --> J
G --> K
G --> L
M --> N
M --> O
M --> P
M --> Q
```

Sources: [train_monomer.sh L1-L54](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer.sh#L1-L54)

#### Multimer Training

```mermaid
flowchart TD

A["train_multimer.sh"]
B["unicore-train"]
C["--task af2"]
D["--loss afm"]
E["--arch af2"]
F["--model-name multimer"]
G["Key Differences"]
H["Loss: afm vs af2"]
I["Different default steps"]
J["Multimer-specific config"]

A --> B
B --> C
B --> D
B --> E
B --> F
G --> H
G --> I
G --> J
```

Sources: [train_multimer.sh L1-L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer.sh#L1-L53)

### Fine-tuning Existing Models

Fine-tuning scripts support continuing training from pre-trained checkpoints:

```mermaid
flowchart TD

I["Lower learning rate"]
J["lr=5e-4 vs 1e-3"]
K["Shorter training"]
L["total_step=10000"]
M["Different decay"]
N["decay_ratio=1.0"]
A["finetune_monomer.sh"]
B["Checkpoint exists?"]
C["Resume from checkpoint"]
D["--finetune-from-model"]
E["--load-from-ema"]
F["finetune_multimer.sh"]
G["Similar logic"]
H["Multimer-specific config"]

subgraph subGraph1 ["Fine-tuning Parameters"]
    I
    J
    K
    L
    M
    N
    I --> J
    K --> L
    M --> N
end

subgraph subGraph0 ["Fine-tuning Flow"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    B --> C
    B --> D
    D --> E
    F --> G
    G --> H
end
```

Sources: [finetune_monomer.sh L37-L43](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_monomer.sh#L37-L43)

 [finetune_multimer.sh L37-L43](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_multimer.sh#L37-L43)

## Training Infrastructure

### Distributed Training Setup

The training system uses PyTorch distributed training with the following architecture:

```mermaid
flowchart TD

K["unicore-train"]
L["--ddp-backend=no_c10d"]
M["--user-dir unifold"]
N["--num-workers 4"]
O["--allreduce-fp32-grad"]
A["torch.distributed.launch"]
B["--nproc_per_node"]
C["--master_port"]
D["--nnodes"]
E["--node_rank"]
F["Environment"]
G["NCCL_ASYNC_ERROR_HANDLING=1"]
H["OMP_NUM_THREADS=1"]
I["OMPI_COMM_WORLD_SIZE"]
J["OMPI_COMM_WORLD_RANK"]

subgraph subGraph1 ["Training Configuration"]
    K
    L
    M
    N
    O
    K --> L
    K --> M
    K --> N
    K --> O
end

subgraph subGraph0 ["Distributed Training"]
    A
    B
    C
    D
    E
    F
    G
    H
    I
    J
    A --> B
    B --> C
    B --> D
    B --> E
    F --> G
    F --> H
    F --> I
    F --> J
end
```

Sources: [train_monomer.sh L15-L51](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer.sh#L15-L51)

 [train_multimer.sh L15-L50](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer.sh#L15-L50)

### Optimization and Performance

Key performance optimizations in the training setup:

| Parameter | Purpose | Value |
| --- | --- | --- |
| `--bf16` | Mixed precision training | Enabled |
| `--bf16-sr` | BF16 state reduction | Enabled |
| `--ema-decay` | Exponential moving average | 0.999 |
| `--per-sample-clip-norm` | Gradient clipping | 0.1 |
| `--data-buffer-size` | Data loading optimization | 32 |

Sources: [train_monomer.sh L51](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer.sh#L51-L51)

 [finetune_monomer.sh L57](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_monomer.sh#L57-L57)

### Learning Rate Scheduling

All training scripts use exponential decay scheduling:

```mermaid
flowchart TD

A["Initial LR"]
B["Warmup Phase"]
C["Exponential Decay"]
D["Parameters"]
E["--lr 1e-3 or 5e-4"]
F["--warmup-updates 1000"]
G["--decay-ratio 0.95"]
H["--decay-steps 50000"]
I["--stair-decay"]

A --> B
B --> C
D --> E
D --> F
D --> G
D --> H
D --> I
```

Sources: [train_monomer.sh L46-L47](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer.sh#L46-L47)

 [finetune_monomer.sh L53](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/finetune_monomer.sh#L53-L53)

## Demo Training Scripts

For testing and experimentation, simplified demo scripts are provided:

### Demo Script Features

```mermaid
flowchart TD

A["train_monomer_demo.sh"]
B["Single GPU setup"]
C["Short training: 1000 steps"]
D["Example data: ./example_data/"]
E["train_multimer_demo.sh"]
F["Multimer configuration"]
G["--model-name multimer"]
H["--loss afm"]
I["Key Differences"]
J["Reduced complexity"]
K["Quick validation"]
L["Educational purposes"]

A --> B
A --> C
A --> D
E --> F
E --> G
E --> H
I --> J
I --> K
I --> L
```

Sources: [train_monomer_demo.sh L7-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer_demo.sh#L7-L15)

 [train_multimer_demo.sh L7-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer_demo.sh#L7-L15)

The demo scripts use the same underlying infrastructure but with reduced training duration and simplified configurations, making them suitable for testing the training pipeline without requiring extensive computational resources.