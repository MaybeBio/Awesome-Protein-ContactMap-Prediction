# DiffusionLitModule Overview

> **Relevant source files**
> * [configs/model/diffusion.yaml](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml)

## Purpose and Scope

This document explains the `DiffusionLitModule` class, which serves as the top-level PyTorch Lightning module in the IDPFold system. `DiffusionLitModule` orchestrates all aspects of the diffusion model including the neural network architecture, diffusion process, training optimization, and inference generation. This page focuses on the module's structure, responsibilities, and how it integrates with the PyTorch Lightning framework.

For details on specific sub-components, see:

* Neural network architecture: [DenoisingNet](/Junjie-Zhu/IDPFold/4.2-denoisingnet)
* Diffusion process mechanics: [FrameDiffuser](/Junjie-Zhu/IDPFold/4.3-framediffuser)
* Loss computation details: [Loss Functions](/Junjie-Zhu/IDPFold/4.4-loss-functions)
* Optimization strategy: [Optimizer and Scheduler](/Junjie-Zhu/IDPFold/4.5-optimizer-and-scheduler)

---

## Module Location and Instantiation

The `DiffusionLitModule` class is defined in `src/models/diffusion_module.py` and is instantiated via Hydra configuration from [configs/model/diffusion.yaml L1](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L1-L1)

```yaml
_target_: src.models.diffusion_module.DiffusionLitModule
```

When IDPFold runs training or inference, Hydra automatically instantiates this module by:

1. Reading the configuration file
2. Resolving the `_target_` path to locate the class
3. Recursively instantiating nested components (net, diffuser, optimizer, scheduler)
4. Passing all configured parameters to the constructor

**Sources:** [configs/model/diffusion.yaml L1](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L1-L1)

---

## Component Architecture

The `DiffusionLitModule` is composed of five primary components, each configured independently in the YAML file:

| Component | Configuration Path | Purpose | Target Class |
| --- | --- | --- | --- |
| **net** | `net` | Neural network that predicts noise to remove from structures | `src.models.net.denoising_ipa.DenoisingNet` |
| **diffuser** | `diffuser` | Manages diffusion process for translations and rotations | `src.models.score.frame.FrameDiffuser` |
| **optimizer** | `optimizer` | Gradient-based parameter optimization | `torch.optim.Adam` |
| **scheduler** | `scheduler` | Learning rate adjustment strategy | `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| **loss** | `loss` | Loss function weights and configurations | (dict configuration) |

**Sources:** [configs/model/diffusion.yaml L3-L85](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L3-L85)

---

## DiffusionLitModule Component Hierarchy

```mermaid
flowchart TD

DLM["DiffusionLitModule<br>(src.models.diffusion_module.DiffusionLitModule)"]
DN["DenoisingNet<br>(src.models.net.denoising_ipa.DenoisingNet)"]
EM["EmbeddingModule<br>(init_embed_size: 32<br>node_embed_size: 256<br>edge_embed_size: 128)"]
TIPA["TranslationIPA<br>(c_s: 256, c_z: 128<br>no_ipa_blocks: 4<br>no_heads: 8)"]
FD["FrameDiffuser<br>(src.models.score.frame.FrameDiffuser)"]
R3D["R3Diffuser<br>(translation in 3D space<br>min_b: 0.1, max_b: 20.0)"]
SO3D["SO3Diffuser<br>(rotation on SO(3) manifold<br>num_omega: 1000<br>num_sigma: 1000)"]
OPT["Adam Optimizer<br>(lr: 1e-4<br>weight_decay: 0.0)"]
SCH["ReduceLROnPlateau<br>(mode: min<br>factor: 0.1<br>patience: 10)"]
L["Loss Configuration"]
LT["translation<br>(weight: 1.0)"]
LR["rotation<br>(weight: 1.0)"]
LB["backbone<br>(weight: 0.25<br>enabled: true)"]
LP["pwd<br>(weight: 0.25<br>enabled: true)"]

DLM --> DN
DLM --> FD
DLM --> OPT
DLM --> SCH
DLM --> L

subgraph subGraph3 ["Loss Functions"]
    L
    LT
    LR
    LB
    LP
    L --> LT
    L --> LR
    L --> LB
    L --> LP
end

subgraph subGraph2 ["Training Components"]
    OPT
    SCH
end

subgraph subGraph1 ["Diffusion Process"]
    FD
    R3D
    SO3D
    FD --> R3D
    FD --> SO3D
end

subgraph subGraph0 ["Neural Network"]
    DN
    EM
    TIPA
    DN --> EM
    DN --> TIPA
end
```

This diagram shows how `DiffusionLitModule` aggregates independently configured components. Each component is instantiated from its `_target_` specification in the YAML configuration and manages a distinct aspect of the model.

**Sources:** [configs/model/diffusion.yaml L16-L85](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L16-L85)

---

## PyTorch Lightning Integration

`DiffusionLitModule` extends `pytorch_lightning.LightningModule`, which provides standardized methods for training and inference. The module implements key Lightning hook methods:


The Lightning framework calls these methods automatically based on the execution mode (`Trainer.fit()` for training, `Trainer.predict()` for inference), abstracting away boilerplate code for distributed training, checkpointing, and logging.

**Sources:** [configs/model/diffusion.yaml L1-L103](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L1-L103)

 system architecture diagrams

---

## Training Mode Configuration

During training, `DiffusionLitModule` uses multiple loss functions with configurable weights to guide the model. The loss configuration controls which objectives are active:

| Loss Component | Enabled | Weight | Threshold | Purpose |
| --- | --- | --- | --- | --- |
| **translation** | Always | 1.0 | x0_threshold: 1.0 | Penalizes errors in predicted 3D translations |
| **rotation** | Always | 1.0 | - | Penalizes errors in predicted rotations on SO(3) |
| **backbone** | true | 0.25 | t_threshold: 0.25 | Enforces realistic backbone geometry |
| **pwd** | true | 0.25 | t_threshold: 0.25 | Maintains pairwise distance distributions |
| **distogram** | false | - | - | Auxiliary loss (disabled) |
| **supervised_chi** | false | - | - | Side chain angle loss (disabled) |
| **lddt** | false | - | - | Local distance difference test (disabled) |
| **fape** | false | - | - | Frame aligned point error (disabled) |
| **tm** | false | - | - | Template modeling score (disabled) |

The `t_threshold` parameters indicate that certain losses (backbone, pwd) are only applied when the diffusion timestep `t` is below the threshold, focusing auxiliary losses on the later stages of denoising where fine details matter.

**Sources:** [configs/model/diffusion.yaml L60-L85](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L60-L85)

---

## Inference Mode Configuration

When generating conformational ensembles, `DiffusionLitModule` operates in inference mode with specific sampling parameters:

```mermaid
flowchart TD

OD["output_dir: paths.output_dir/samples"]
BO["backward_only: true"]
NR["n_replica: 192"]
NT["num_timesteps: 1000"]
NS["noise_scale: 1.0"]
SC["self_conditioning: true"]
MT["min_t: 0.01"]
PF["probability_flow: false"]
DMIN["delta_min: 0.25"]
DMAX["delta_max: 0.35"]
DSTEP["delta_step: 0.05"]
RPB["replica_per_batch: 64"]

NR --> RPB
DSTEP --> NR

subgraph Batching ["Batching"]
    RPB
end

subgraph subGraph1 ["Delta Sweep"]
    DMIN
    DMAX
    DSTEP
    DMIN --> DMAX
    DMAX --> DSTEP
end

subgraph subGraph0 ["Inference Parameters"]
    NR
    NT
    NS
    SC
    MT
    PF
    NT --> NR
    NS --> NT
    SC --> NT
end

subgraph Output ["Output"]
    OD
    BO
end
```

**Key Inference Parameters:**

* **num_timesteps (1000)**: Number of denoising steps from pure noise to final structure
* **n_replica (192)**: Number of independent structure samples to generate per sequence
* **noise_scale (1.0)**: Scaling factor for initial noise magnitude
* **self_conditioning (true)**: Uses previous prediction to condition current denoising step, improving consistency
* **delta_min/max/step**: Controls sweep over different noise schedules, generating multiple ensembles per protein
* **replica_per_batch (64)**: Batches replicas for memory efficiency during GPU inference
* **min_t (0.01)**: Minimum timestep value to prevent numerical instability at t=0

The total number of generated structures is: `n_replica × ((delta_max - delta_min) / delta_step + 1) = 192 × 3 = 576` structures per protein.

**Sources:** [configs/model/diffusion.yaml L87-L101](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L87-L101)

---

## Optimizer and Scheduler Configuration

```mermaid
flowchart TD

DLM["DiffusionLitModule"]
ADAM["torch.optim.Adam"]
LR["learning rate: 1e-4"]
WD["weight_decay: 0.0"]
SCHED["torch.optim.lr_scheduler.ReduceLROnPlateau"]
MODE["mode: min<br>(monitors validation loss)"]
FACTOR["factor: 0.1<br>(reduces lr by 10x)"]
PATIENCE["patience: 10<br>(waits 10 epochs)"]

DLM --> ADAM
DLM --> SCHED
SCHED --> LR

subgraph subGraph1 ["Scheduler: ReduceLROnPlateau"]
    SCHED
    MODE
    FACTOR
    PATIENCE
    SCHED --> MODE
    SCHED --> FACTOR
    SCHED --> PATIENCE
end

subgraph subGraph0 ["Optimizer: Adam"]
    ADAM
    LR
    WD
    ADAM --> LR
    ADAM --> WD
end
```

The module uses the Adam optimizer with a fixed learning rate of `1e-4` and no weight decay (L2 regularization disabled). The `ReduceLROnPlateau` scheduler monitors validation loss and reduces the learning rate by a factor of 10 if no improvement is observed for 10 consecutive epochs. This adaptive scheduling helps the model converge to better solutions by taking smaller steps when approaching local minima.

The `_partial_: true` flag in the configuration [configs/model/diffusion.yaml L5-L11](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L5-L11)

 indicates that these objects are created as partial functions, with the model parameters being passed later when `configure_optimizers()` is called by PyTorch Lightning.

**Sources:** [configs/model/diffusion.yaml L3-L14](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L3-L14)

---

## Self-Conditioning Mechanism

The `self_conditioning` parameter appears in two locations in the configuration:

1. **EmbeddingModule** [configs/model/diffusion.yaml L26](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L26-L26) : `self_conditioning: true`
2. **Inference settings** [configs/model/diffusion.yaml L97](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L97-L97) : `self_conditioning: true`

This indicates that `DiffusionLitModule` implements self-conditioning, a technique where the model's previous prediction is fed back as an additional input to the current denoising step. This creates a feedback loop that helps the model produce more consistent predictions across timesteps.

For detailed explanation of the self-conditioning mechanism, see [Self-Conditioning](/Junjie-Zhu/IDPFold/7.4-self-conditioning).

**Sources:** [configs/model/diffusion.yaml L26-L97](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L26-L97)

---

## Model Compilation

The configuration includes a `compile` flag:

```yaml
compile: false
```

When set to `true`, this would enable PyTorch 2.0's `torch.compile()` feature, which uses TorchDynamo to JIT-compile the model for faster execution. The flag is currently disabled, likely for debugging purposes or compatibility with the current PyTorch version.

**Sources:** [configs/model/diffusion.yaml L103](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L103-L103)

---

## Integration with System Workflow

```mermaid
flowchart TD

CONFIG["configs/model/diffusion.yaml"]
HYDRA["Hydra Framework"]
DLM["DiffusionLitModule<br>(instantiated)"]
TRAIN["train.py"]
TRAIN_DATA["Training DataModule"]
TRAIN_LOOP["training_step()"]
EVAL["eval.py"]
EVAL_DATA["Test DataModule<br>(preprocessed embeddings)"]
PREDICT["predict_step()"]
ENSEMBLES["Conformational Ensembles<br>(output structures)"]
CKPT["Model Checkpoint<br>(Google Drive)"]

CONFIG --> HYDRA
HYDRA --> DLM
DLM --> TRAIN_LOOP
CKPT --> DLM
DLM --> PREDICT
EVAL --> DLM
TRAIN --> DLM

subgraph subGraph1 ["Inference Path"]
    EVAL
    EVAL_DATA
    PREDICT
    ENSEMBLES
    EVAL_DATA --> PREDICT
    PREDICT --> ENSEMBLES
end

subgraph subGraph0 ["Training Path (incomplete)"]
    TRAIN
    TRAIN_DATA
    TRAIN_LOOP
    TRAIN_DATA --> TRAIN_LOOP
end
```

The `DiffusionLitModule` serves as the central model component in both training and inference workflows:

* **Training**: Would receive batches from a training DataModule, compute losses via `training_step()`, and update parameters via the optimizer
* **Inference**: Loads weights from a checkpoint, receives preprocessed embeddings from the test DataModule, and generates structure ensembles via `predict_step()`

The Hydra configuration system ensures that the exact same model architecture used during training can be reconstructed for inference by loading the same `diffusion.yaml` configuration.

**Sources:** [configs/model/diffusion.yaml L1-L103](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L1-L103)

 system architecture diagrams

---

## Summary

`DiffusionLitModule` is the top-level orchestrator of the IDPFold diffusion model. It:

1. **Aggregates** all model components (network, diffuser, optimizer, scheduler, losses)
2. **Inherits** from PyTorch Lightning's `LightningModule` for standardized training/inference
3. **Configures** behavior via the Hydra configuration system
4. **Implements** training logic through loss computation and optimization
5. **Generates** conformational ensembles through iterative denoising during inference
6. **Manages** self-conditioning, noise schedules, and sampling strategies

The modular design allows researchers to independently modify each component (e.g., swap the neural architecture, change the diffusion schedule, adjust loss weights) by editing the configuration file without modifying the `DiffusionLitModule` code itself.

**Sources:** [configs/model/diffusion.yaml L1-L103](https://github.com/Junjie-Zhu/IDPFold/blob/ba40f40c/configs/model/diffusion.yaml#L1-L103)

 system architecture diagrams