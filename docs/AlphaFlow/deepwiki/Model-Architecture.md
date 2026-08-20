# Model Architecture

> **Relevant source files**
> * [alphaflow/model/esmfold.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py)
> * [alphaflow/model/wrapper.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py)

This document provides detailed technical specifications of the core model architectures and training frameworks in AlphaFlow. It covers the `ESMFold` and `AlphaFold` model implementations, the `ModelWrapper` training system, and the diffusion-based structure generation process.

For information about using these models for inference, see [Inference System](/bjing2016/alphaflow/3-inference-system). For training these models, see [Training System](/bjing2016/alphaflow/4-training-system).

## Core Model Components

The AlphaFlow system is built around two primary model architectures: `ESMFold` and `AlphaFold`, both modified to support diffusion-based ensemble generation. The models share a common diffusion framework but differ in their sequence processing approaches.

```mermaid
flowchart TD

ESM["ESM Language Model<br>esm.pretrained"]
ESM_MLP["esm_s_mlp<br>LayerNorm + Linear"]
ESM_COMBINE["esm_s_combine<br>Parameter Weights"]
EMBED["embedding<br>nn.Embedding"]
TRUNK["trunk<br>FoldingTrunk"]
DIST_HEAD["distogram_head<br>nn.Linear"]
LDDT_HEAD["lddt_head<br>PerResidueLDDTCaPredictor"]
INPUT_PAIR["input_pair_embedding<br>Linear"]
TIME_PROJ["input_time_projection<br>GaussianFourierProjection"]
TIME_EMBED["input_time_embedding<br>Linear"]
PAIR_STACK["input_pair_stack<br>InputPairStack"]
EXTRA_PAIR["extra_input_pair_embedding<br>Linear"]
EXTRA_STACK["extra_input_pair_stack<br>InputPairStack"]

ESM_MLP --> EMBED
PAIR_STACK --> TRUNK
EXTRA_STACK --> PAIR_STACK

subgraph subGraph3 ["Optional Components"]
    EXTRA_PAIR
    EXTRA_STACK
    EXTRA_PAIR --> EXTRA_STACK
end

subgraph subGraph2 ["Diffusion Components"]
    INPUT_PAIR
    TIME_PROJ
    TIME_EMBED
    PAIR_STACK
    INPUT_PAIR --> PAIR_STACK
    TIME_PROJ --> TIME_EMBED
    TIME_EMBED --> PAIR_STACK
end

subgraph subGraph1 ["Common Components"]
    EMBED
    TRUNK
    DIST_HEAD
    LDDT_HEAD
    EMBED --> TRUNK
    TRUNK --> DIST_HEAD
    TRUNK --> LDDT_HEAD
end

subgraph subGraph0 ["ESMFold Architecture"]
    ESM
    ESM_MLP
    ESM_COMBINE
    ESM --> ESM_COMBINE
    ESM_COMBINE --> ESM_MLP
end
```

**ESMFold Model Architecture**

Sources: [alphaflow/model/esmfold.py L49-L437](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L49-L437)

The `ESMFold` class integrates Meta's ESM language model with AlphaFold's structure prediction framework. Key architectural components include:

| Component | Class/Function | Purpose |
| --- | --- | --- |
| `esm` | `esm.pretrained` models | Pretrained language model backbone |
| `esm_s_combine` | `nn.Parameter` | Learnable layer combination weights |
| `esm_s_mlp` | `nn.Sequential` | Maps ESM features to structure space |
| `trunk` | `FoldingTrunk` | Core structure prediction module |
| `input_pair_embedding` | `Linear` | Embeds distance information |
| `input_time_projection` | `GaussianFourierProjection` | Time encoding for diffusion |

The model supports multiple ESM variants through the `esm_registry` [alphaflow/model/esmfold.py L34-L46](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L34-L46)

 including models from 8M to 15B parameters.

## Training Framework Architecture

The training system is built around the `ModelWrapper` base class with specialized implementations for different model types.

```mermaid
flowchart TD

BASE["ModelWrapper<br>pl.LightningModule"]
HARMONIC["harmonic_prior<br>HarmonicPrior"]
LOSS["loss<br>AlphaFoldLoss"]
EMA["ema<br>ExponentialMovingAverage"]
ESM_WRAP["ESMFoldWrapper"]
ESM_MODEL["model<br>ESMFold"]
ESM_TEACHER["teacher<br>ESMFold"]
AF_WRAP["AlphaFoldWrapper"]
AF_MODEL["model<br>AlphaFold"]
AF_TEACHER["teacher<br>AlphaFold"]
TRAIN_STEP["training_step()"]
DISTILL_STEP["disillation_training_step()"]
INFERENCE["inference()"]
ADD_NOISE["_add_noise()"]
VAL_STEP["validation_step()"]
COMPUTE_METRICS["_compute_validation_metrics()"]
PROTEIN_METRICS["protein.global_metrics()"]

BASE --> ESM_WRAP
BASE --> AF_WRAP
BASE --> TRAIN_STEP
BASE --> DISTILL_STEP
BASE --> INFERENCE
BASE --> ADD_NOISE
BASE --> VAL_STEP
HARMONIC --> ADD_NOISE
LOSS --> TRAIN_STEP

subgraph subGraph4 ["Validation & Metrics"]
    VAL_STEP
    COMPUTE_METRICS
    PROTEIN_METRICS
    VAL_STEP --> COMPUTE_METRICS
    VAL_STEP --> PROTEIN_METRICS
end

subgraph subGraph3 ["Training Methods"]
    TRAIN_STEP
    DISTILL_STEP
    INFERENCE
    ADD_NOISE
end

subgraph subGraph2 ["AlphaFoldWrapper"]
    AF_WRAP
    AF_MODEL
    AF_TEACHER
    AF_WRAP --> AF_MODEL
    AF_WRAP --> AF_TEACHER
end

subgraph ESMFoldWrapper ["ESMFoldWrapper"]
    ESM_WRAP
    ESM_MODEL
    ESM_TEACHER
    ESM_WRAP --> ESM_MODEL
    ESM_WRAP --> ESM_TEACHER
end

subgraph subGraph0 ["ModelWrapper Base Class"]
    BASE
    HARMONIC
    LOSS
    EMA
    EMA --> BASE
end
```

**Training Framework Components**

Sources: [alphaflow/model/wrapper.py L51-L515](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L51-L515)

The `ModelWrapper` class [alphaflow/model/wrapper.py L51](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L51-L51)

 provides the core training infrastructure:

| Method | Purpose | Key Features |
| --- | --- | --- |
| `training_step()` | Standard training loop | Noise injection, self-conditioning, loss computation |
| `disillation_training_step()` | Teacher-student distillation | Multi-step teacher inference, student training |
| `inference()` | Diffusion-based sampling | Iterative denoising with configurable schedules |
| `_add_noise()` | Noise injection | RMSD alignment, harmonic prior sampling |

The system supports two specialized wrappers:

* `ESMFoldWrapper` [alphaflow/model/wrapper.py L466-L489](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L466-L489)  for ESM-based models
* `AlphaFoldWrapper` [alphaflow/model/wrapper.py L491-L515](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L491-L515)  for AlphaFold-based models

## Diffusion Process Architecture

The diffusion framework enables ensemble generation through iterative denoising of protein structures.

```mermaid
flowchart TD

HARMONIC["HarmonicPrior.sample()"]
RMSD_ALIGN["rmsdalign()"]
INTERPOLATE["Linear Interpolation<br>(1-t)clean + tnoisy"]
TIME_T["t ~ Uniform(0,1)"]
GAUSS_PROJ["GaussianFourierProjection"]
TIME_LINEAR["Linear Layer"]
PAIR_EMBED["Pair Distance Embedding"]
TIME_COND["Time Conditioning"]
STRUCTURE_PRED["Structure Prediction"]
SCHEDULE["[1.0, 0.75, 0.5, 0.25, 0.1, 0]"]
ITERATIVE["Iterative Denoising"]
SELF_COND["Self Conditioning"]

INTERPOLATE --> PAIR_EMBED
TIME_LINEAR --> TIME_COND
STRUCTURE_PRED --> ITERATIVE

subgraph subGraph3 ["Inference Schedule"]
    SCHEDULE
    ITERATIVE
    SELF_COND
    SCHEDULE --> ITERATIVE
    ITERATIVE --> SELF_COND
end

subgraph subGraph2 ["Model Forward Pass"]
    PAIR_EMBED
    TIME_COND
    STRUCTURE_PRED
    PAIR_EMBED --> STRUCTURE_PRED
    TIME_COND --> STRUCTURE_PRED
end

subgraph subGraph1 ["Time Embedding"]
    TIME_T
    GAUSS_PROJ
    TIME_LINEAR
    TIME_T --> GAUSS_PROJ
    GAUSS_PROJ --> TIME_LINEAR
end

subgraph subGraph0 ["Noise Injection Process"]
    HARMONIC
    RMSD_ALIGN
    INTERPOLATE
    HARMONIC --> RMSD_ALIGN
    RMSD_ALIGN --> INTERPOLATE
end
```

**Diffusion Implementation Details**

Sources: [alphaflow/model/esmfold.py L82-L92](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L82-L92)

 [alphaflow/model/wrapper.py L52-L71](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L52-L71)

The diffusion process operates on pseudo-beta carbon distances rather than raw coordinates:

1. **Noise Addition** [alphaflow/model/wrapper.py L52-L71](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L52-L71) : * Sample from `HarmonicPrior` * RMSD align noisy structure to clean structure * Linear interpolation with time parameter `t`
2. **Time Conditioning** [alphaflow/model/esmfold.py L82-L89](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L82-L89) : * `GaussianFourierProjection` encodes time `t` * Linear embedding to pair representation dimension * Added to pair embeddings before trunk processing
3. **Inference Schedule** [alphaflow/model/wrapper.py L369-L383](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L369-L383) : * Default schedule: `[1.0, 0.75, 0.5, 0.25, 0.1, 0]` * Iterative denoising with optional self-conditioning * RMSD alignment between steps

## Model Variants and Extensions

The architecture supports several specialized variants through conditional inputs and training strategies.

```mermaid
flowchart TD

ESM_BASE["ESMFold Base"]
AF_BASE["AlphaFold Base"]
PDB_TRAIN["PDB Training<br>Experimental Structures"]
MD_TRAIN["MD Training<br>ATLAS Trajectories"]
TEMPLATE_TRAIN["Template Training<br>Reference Structures"]
EXTRA_INPUT["extra_input=True<br>Template Conditioning"]
DISTILLATION["Distillation Mode<br>Teacher-Student"]
SELF_COND["Self Conditioning<br>prev_outputs"]
ESM_48L["ESMFlow 48-layer"]
ESM_12L["ESMFlow 12-layer"]
AF_48L["AlphaFlow 48-layer"]
AF_12L["AlphaFlow 12-layer"]

ESM_BASE --> PDB_TRAIN
AF_BASE --> PDB_TRAIN
ESM_BASE --> EXTRA_INPUT
AF_BASE --> EXTRA_INPUT
TEMPLATE_TRAIN --> EXTRA_INPUT
ESM_BASE --> DISTILLATION
AF_BASE --> DISTILLATION
ESM_BASE --> ESM_48L
AF_BASE --> AF_48L

subgraph subGraph3 ["Model Configurations"]
    ESM_48L
    ESM_12L
    AF_48L
    AF_12L
    ESM_48L --> ESM_12L
    AF_48L --> AF_12L
end

subgraph subGraph2 ["Architectural Extensions"]
    EXTRA_INPUT
    DISTILLATION
    SELF_COND
end

subgraph subGraph1 ["Training Variants"]
    PDB_TRAIN
    MD_TRAIN
    TEMPLATE_TRAIN
    PDB_TRAIN --> MD_TRAIN
    MD_TRAIN --> TEMPLATE_TRAIN
end

subgraph subGraph0 ["Base Models"]
    ESM_BASE
    AF_BASE
end
```

**Extended Architecture Features**

Sources: [alphaflow/model/esmfold.py L94-L101](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L94-L101)

 [alphaflow/model/wrapper.py L249-L264](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L249-L264)

| Feature | Implementation | Purpose |
| --- | --- | --- |
| `extra_input` | Additional pair embeddings [alphaflow/model/esmfold.py L94-L101](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L94-L101) | Template/reference structure conditioning |
| Distillation | Teacher-student training [alphaflow/model/wrapper.py L249-L264](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L249-L264) | Model compression and knowledge transfer |
| Self-conditioning | Previous output recycling [alphaflow/model/esmfold.py L284-L297](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L284-L297) | Improved structure refinement |
| Layer variants | 12-layer vs 48-layer models | Speed-accuracy tradeoffs |

The `extra_input` mode allows conditioning on reference structures through `extra_all_atom_positions` in the batch, enabling template-guided structure prediction.

Sources: [alphaflow/model/esmfold.py L1-L437](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L1-L437)

 [alphaflow/model/wrapper.py L1-L515](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L1-L515)