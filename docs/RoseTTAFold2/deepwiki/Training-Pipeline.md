# Training Pipeline

> **Relevant source files**
> * [network/loss.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/loss.py)
> * [network/train_multi_deep.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py)

This document describes the distributed training infrastructure for RoseTTAFold2, including the main training loop, data management, loss functions, and validation procedures. The training pipeline supports multi-GPU distributed training with various data types and implements advanced features like exponential moving averages and recycling iterations.

For information about the neural network architecture being trained, see [Core Architecture](/uw-ipd/RoseTTAFold2/3-core-architecture). For details about data loading and preprocessing, see [Data Loading for Training](/uw-ipd/RoseTTAFold2/5.2-data-loading-for-training).

## Training Architecture Overview

The training pipeline is built around a distributed training framework that handles multiple data types and implements sophisticated training strategies including recycling and validation across different protein structure prediction tasks.

**Training System Architecture**

```mermaid
flowchart TD

A["train_multi_deep.py"]
B["get_args()"]
C["Trainer Class"]
D["run_model_training()"]
E["train_model()"]
F["EMA Class"]
G["RoseTTAFoldModule"]
H["DDP Wrapper"]
I["DistilledDataset"]
J["Dataset"]
K["DatasetComplex"]
L["DistributedWeightedSampler"]
M["train_cycle()"]
N["valid_pdb_cycle()"]
O["valid_ppi_cycle()"]
P["calc_loss()"]

B --> C
E --> F
E --> G
E --> I
E --> J
E --> K
E --> M
E --> N
E --> O

subgraph subGraph4 ["Training Execution"]
    M
    N
    O
    P
    M --> P
end

subgraph subGraph3 ["Data Management"]
    I
    J
    K
    L
    I --> L
end

subgraph subGraph2 ["Model Management"]
    F
    G
    H
    G --> H
end

subgraph subGraph1 ["Training Coordination"]
    C
    D
    E
    C --> D
    D --> E
end

subgraph subGraph0 ["Entry Point"]
    A
    B
    A --> B
end
```

Sources: [network/train_multi_deep.py L1-L1184](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L1-L1184)

## Trainer Class and Core Components

The `Trainer` class serves as the central coordinator for the training process, managing model initialization, data loading, and training execution.

**Core Training Components**

```mermaid
flowchart TD

A["init()"]
B["model_param"]
C["loader_param"]
D["loss_param"]
E["EMA"]
F["RoseTTAFoldModule"]
G["DDP"]
H["shadow model"]
I["main model"]
J["AdamW optimizer"]
K["get_stepwise_decay_schedule_with_warmup"]
L["GradScaler"]
M["XYZConverter"]
N["loss_fn CrossEntropyLoss"]
O["active_fn Softmax"]
P["calc_loss()"]

A --> E
A --> J
A --> K
A --> L
A --> M
A --> N
A --> O
A --> P

subgraph subGraph3 ["Loss Components"]
    N
    O
    P
end

subgraph subGraph2 ["Training Infrastructure"]
    J
    K
    L
    M
end

subgraph subGraph1 ["Model Components"]
    E
    F
    G
    H
    I
    E --> F
    E --> H
    E --> I
    F --> G
end

subgraph subGraph0 ["Trainer Class"]
    A
    B
    C
    D
    A --> B
    A --> C
    A --> D
end
```

The `Trainer` class is initialized with comprehensive parameters for model architecture, data loading, and loss computation:

| Parameter Category | Key Components | Purpose |
| --- | --- | --- |
| `model_param` | Network architecture settings | Configures `RoseTTAFoldModule` |
| `loader_param` | Data loading parameters | Controls dataset behavior |
| `loss_param` | Loss function weights | Balances different loss terms |
| Training Settings | `batch_size`, `accum_step`, `maxcycle` | Controls training dynamics |

Sources: [network/train_multi_deep.py L104-L149](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L104-L149)

## Data Loading and Management

The training pipeline supports multiple data types through specialized dataset classes and implements sophisticated sampling strategies.

**Data Loading Architecture**

```mermaid
flowchart TD

A["PDB structures"]
B["FoldingBank FB"]
C["Complex structures"]
D["Negative examples"]
E["Dataset"]
F["DatasetComplex"]
G["DistilledDataset"]
H["loader_pdb"]
I["loader_fb"]
J["loader_complex"]
K["DistributedWeightedSampler"]
L["train_sampler"]
M["valid_pdb_sampler"]
N["valid_compl_sampler"]

A --> H
B --> I
C --> J
D --> J
H --> E
I --> G
J --> F
J --> G
E --> K
F --> K
G --> L

subgraph subGraph3 ["Sampling Strategy"]
    K
    L
    M
    N
    K --> M
    K --> N
end

subgraph subGraph2 ["Data Loaders"]
    H
    I
    J
end

subgraph subGraph1 ["Dataset Classes"]
    E
    F
    G
end

subgraph subGraph0 ["Data Sources"]
    A
    B
    C
    D
end
```

The data management system handles different validation sets:

* **PDB validation**: Standard protein structures
* **Homo validation**: Homo-oligomeric structures
* **Complex validation**: Protein-protein complexes
* **Negative validation**: Non-interacting protein pairs

Sources: [network/train_multi_deep.py L421-L467](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L421-L467)

## Training Loop and Recycling

The training process implements a sophisticated recycling mechanism where predictions from previous iterations are used as input for subsequent iterations.

**Training Cycle Flow**

```mermaid
flowchart TD

A["train_cycle()"]
B["_prepare_input()"]
C["N_cycle = random(1, maxcycle)"]
D["i_cycle = 0"]
E["_get_model_input()"]
F["ddp_model(**input_i)"]
G["i_cycle < N_cycle-1?"]
H["torch.no_grad()"]
I["return_raw=True"]
J["Final iteration"]
K["return_raw=False"]
L["_get_loss_and_misc()"]
M["calc_loss()"]
N["Backward pass"]
O["EMA update"]

C --> D
K --> L

subgraph subGraph2 ["Loss Calculation"]
    L
    M
    N
    O
    L --> M
    M --> N
    N --> O
end

subgraph subGraph1 ["Recycling Loop"]
    D
    E
    F
    G
    H
    I
    J
    K
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> D
    G --> J
    J --> K
end

subgraph subGraph0 ["Training Iteration"]
    A
    B
    C
    A --> B
    B --> C
end
```

Key training features:

* **Random recycling**: Number of cycles varies from 1 to `maxcycle`
* **Gradient accumulation**: Supports `ACCUM_STEP` for effective batch size scaling
* **Mixed precision**: Uses `torch.cuda.amp.autocast` with bfloat16
* **Memory optimization**: Intermediate cycles use `torch.no_grad()` and `return_raw=True`

Sources: [network/train_multi_deep.py L752-L881](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L752-L881)

## Loss Functions and Validation

The training pipeline implements a comprehensive loss function that combines multiple structural and sequence-based terms.

**Loss Function Components**

```mermaid
flowchart TD

A["c6d_loss"]
B["masked_token_loss"]
C["binder_loss"]
D["experimental_loss"]
E["FAPE_loss"]
F["torsion_loss"]
G["allatom_FAPE"]
H["lddt_loss"]
I["bond_length_loss"]
J["bond_angle_loss"]
K["lj_potential"]
L["hbond_loss"]
M["calc_loss()"]
N["calc_str_loss()"]
O["calc_allatom_lddt_w_loss()"]
P["calc_BB_bond_geom()"]

A --> M
B --> M
C --> M
D --> M
E --> N
F --> N
G --> O
H --> O
I --> P
J --> P
K --> M
L --> M

subgraph subGraph3 ["Loss Calculation"]
    M
    N
    O
    P
    M --> N
    M --> O
    M --> P
end

subgraph subGraph2 ["Geometry Loss Terms"]
    I
    J
    K
    L
end

subgraph subGraph1 ["Structural Loss Terms"]
    E
    F
    G
    H
end

subgraph subGraph0 ["Primary Loss Terms"]
    A
    B
    C
    D
end
```

The loss function weights are configurable through `loss_param`:

| Loss Component | Weight Parameter | Purpose |
| --- | --- | --- |
| Distance prediction | `w_dist` | 6D distance/orientation |
| Amino acid prediction | `w_aa` | Masked language modeling |
| Structure prediction | `w_str` | FAPE and torsion angles |
| Binding prediction | `w_bind` | Protein-protein interaction |
| Confidence prediction | `w_lddt`, `w_pae` | Quality estimation |

Sources: [network/train_multi_deep.py L150-L332](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L150-L332)

 [network/loss.py L1-L678](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/loss.py#L1-L678)

## Model Checkpointing and EMA

The training system implements Exponential Moving Average (EMA) for model weights and comprehensive checkpointing.

**EMA and Checkpointing System**

```mermaid
flowchart TD

A["EMA Class"]
B["model (training)"]
C["shadow (inference)"]
D["update()"]
E["decay=0.99"]
F["checkpoint_fn()"]
G["best model"]
H["last model"]
I["load_model()"]
J["model_state_dict"]
K["optimizer_state_dict"]
L["scheduler_state_dict"]
M["scaler_state_dict"]

I --> J
I --> K
I --> L
I --> M

subgraph subGraph2 ["Model States"]
    J
    K
    L
    M
end

subgraph Checkpointing ["Checkpointing"]
    F
    G
    H
    I
    F --> G
    F --> H
    G --> I
    H --> I
end

subgraph subGraph0 ["EMA Implementation"]
    A
    B
    C
    D
    E
    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
end
```

The EMA mechanism maintains two copies of the model:

* **Training model**: Used for gradient computation and parameter updates
* **Shadow model**: Exponentially averaged weights used for inference and validation

The checkpoint system saves:

* Best model based on validation loss
* Most recent model for training resumption
* Complete optimizer and scheduler states
* Training and validation metrics

Sources: [network/train_multi_deep.py L60-L103](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L60-L103)

 [network/train_multi_deep.py L365-L542](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L365-L542)

## Distributed Training Setup

The training pipeline supports both interactive and SLURM-based distributed training across multiple GPUs.

**Distributed Training Architecture**

```mermaid
flowchart TD

A["Interactive Mode"]
B["SLURM Mode"]
C["mp.spawn()"]
D["SLURM_PROCID"]
E["dist.init_process_group()"]
F["nccl backend"]
G["world_size"]
H["rank"]
I["torch.cuda.set_device()"]
J["DDP wrapper"]
K["device_ids=[gpu]"]
L["find_unused_parameters=False"]
M["DistributedSampler"]
N["DistributedWeightedSampler"]
O["set_epoch()"]
P["N_EXAMPLE_PER_EPOCH"]

C --> E
D --> E
E --> I
E --> M
E --> N

subgraph subGraph3 ["Data Distribution"]
    M
    N
    O
    P
    M --> O
    N --> O
    O --> P
end

subgraph subGraph2 ["GPU Management"]
    I
    J
    K
    L
    I --> J
    J --> K
    J --> L
end

subgraph subGraph1 ["DDP Setup"]
    E
    F
    G
    H
    E --> F
    E --> G
    E --> H
end

subgraph subGraph0 ["Launch Methods"]
    A
    B
    C
    D
    A --> C
    B --> D
end
```

Key distributed training features:

* **Automatic GPU detection**: Uses `torch.cuda.device_count()` for interactive mode
* **SLURM integration**: Reads environment variables for cluster deployment
* **Synchronized sampling**: Ensures consistent data distribution across processes
* **Gradient synchronization**: Uses DDP for efficient gradient averaging

The system handles both training and validation data distribution, ensuring each process receives a balanced subset of the total dataset.

Sources: [network/train_multi_deep.py L398-L544](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/train_multi_deep.py#L398-L544)