---
title: "Training System"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/6-training-system
---
# Training System

# Training System

> **Relevant source files**
> - [docs/kernels\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/kernels.md?plain=1)
> - [finetune\_demo\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/finetune_demo.sh)
> - [protenix/utils/permutation/permutation\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/permutation/permutation.py)
> - [protenix/utils/training\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py)
> - [runner/train\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py)
> - [train\_demo\.sh](https://github.com/bytedance/Protenix/blob/c3bfc365/train_demo.sh)

 This document describes the Protenix training system for training and fine\-tuning biomolecular structure prediction models from scratch or from checkpoints\. The system is orchestrated by the `AF3Trainer` class in [runner/train\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py) and supports distributed training with PyTorch DDP, mixed precision training \(BF16/FP32\), gradient accumulation, exponential moving average \(EMA\) weight tracking, and symmetric permutation handling for molecular symmetries\.

## Overview

 The Protenix training system is implemented in [train\.py L54-L428](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L54-L428) with the `AF3Trainer` class as the central orchestrator\. The training pipeline handles:

 - **Model Initialization**: Creates `Protenix` model [protenix\.py L30-L103](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/protenix.py#L30-L103) wraps in DDP for distributed training\.
- **Data Pipeline**: Loads training data via `WeightedMultiDataset` with spatial cropping and weighted sampling [dataloader\.py L28-L112](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataloader.py#L28-L112)
- **Loss Computation**: Calculates loss using `ProtenixLoss` with diffusion, distogram, and confidence terms [loss\.py L33-L228](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/loss.py#L33-L228)
- **Symmetric Permutation**: Aligns predictions to ground truth using `SymmetricPermutation` for chain and atom symmetries [permutation\.py L22-L488](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/permutation/permutation.py#L22-L488)
- **Optimization**: Updates parameters with AdamW optimizer [training\.py L21-L71](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py#L21-L71) and learning rate schedulers [lr\_scheduler\.py L31-L48](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/lr_scheduler.py#L31-L48)
- **EMA Tracking**: Maintains exponential moving average of weights via `EMAWrapper` [ema\.py L21-L36](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/ema.py#L21-L36)
- **Checkpointing**: Saves model, optimizer, and scheduler states periodically [train\.py L226-L240](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L226-L240)

### Entry Points

 The training system is invoked through two main scripts:

| Script | Purpose | Key Parameters |
| --- | --- | --- |
| train\_demo\.sh | Train from scratch | \-\-model\_name, \-\-max\_steps, \-\-train\_crop\_size |
| finetune\_demo\.sh | Fine\-tune from checkpoint | \-\-load\_checkpoint\_path, \-\-load\_ema\_checkpoint\_path, \-\-data\.\{dataset\}\.base\_info\.pdb\_list |

 Both scripts invoke [runner/train\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py) with appropriate configuration overrides\.

 **Diagram: Training System Architecture and Code Components**

  Sources: [train\.py L54-L428](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L54-L428) [train\_demo\.sh L24-L47](https://github.com/bytedance/Protenix/blob/c3bfc365/train_demo.sh#L24-L47) [finetune\_demo\.sh L26-L51](https://github.com/bytedance/Protenix/blob/c3bfc365/finetune_demo.sh#L26-L51) [dataloader\.py L28-L112](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/pipeline/dataloader.py#L28-L112) [training\.py L73-L115](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py#L73-L115) [lr\_scheduler\.py L31-L48](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/lr_scheduler.py#L31-L48)

## AF3Trainer Class

 The `AF3Trainer` class [train\.py L54-L428](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L54-L428) orchestrates all aspects of training and evaluation\.

### Initialization Methods

 The trainer initialization involves multiple setup stages:

| Method | Purpose | Key Actions |
| --- | --- | --- |
| init\_env\(\) | Environment setup | DDP initialization, CUDA setup, seed setting runner/train\.py129\-178 |
| init\_basics\(\) | Basic attributes | Step counters, directory creation, config saving runner/train\.py72\-115 |
| init\_log\(\) | Logging setup | W&B initialization, metric aggregators runner/train\.py116\-127 |
| init\_model\(\) | Model setup | Model creation, DDP wrapping, optimizer, scheduler runner/train\.py157\-201 |
| init\_loss\(\) | Loss components | ProtenixLoss, SymmetricPermutation, LDDTMetrics runner/train\.py179\-191 |
| init\_data\(\) | Data loading | Train/test dataloaders via get\_dataloaders\(\) runner/train\.py218\-224 |

 Diagram: **AF3Trainer Initialization Sequence**

  Sources: [train\.py L62-L71](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L62-L71) [train\.py L72-L115](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L72-L115) [train\.py L129-L178](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L129-L178) [train\.py L157-L201](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L157-L201) [train\.py L218-L224](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L218-L224)

## Loss Calculation and Symmetric Permutation

### Loss Components

 The `ProtenixLoss` class [loss\.py L33-L228](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/loss.py#L33-L228) calculates multiple loss terms during training:

 - **Diffusion loss**: MSE loss on denoising predictions\.
- **Distogram loss**: Cross\-entropy loss on predicted distance distributions\.
- **Confidence loss**: Cross\-entropy loss on confidence metrics \(pLDDT, PAE, PDE\)\.

 The loss calculation involves mini\-rollout permutation managed by `AF3Trainer` [train\.py L314-L324](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L314-L324): it aligns ground truth labels to mini\-rollout predictions using `SymmetricPermutation.permute_label_to_match_mini_rollout()`\.

### Symmetric Permutation System

 The `SymmetricPermutation` class [permutation\.py L22-L488](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/permutation/permutation.py#L22-L488) handles molecular symmetries:

| Method | Purpose | Stage |
| --- | --- | --- |
| permute\_label\_to\_match\_mini\_rollout\(\) | Align labels to mini\-rollout | Training mini\-rollout protenix/utils/permutation/permutation\.py40\-113 |
| permute\_diffusion\_sample\_to\_match\_label\(\) | Align predictions to labels | Training/evaluation full diffusion protenix/utils/permutation/permutation\.py115\-241 |
| permute\_heads\(\) | Permute confidence head outputs | Post\-diffusion protenix/utils/permutation/permutation\.py243\-346 |

 Sources: [permutation\.py L22-L488](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/permutation/permutation.py#L22-L488) [train\.py L314-L324](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L314-L324) [loss\.py L33-L228](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/loss.py#L33-L228)

## Checkpoint Management and EMA

### Checkpoint Loading

 The `try_load_checkpoint()` method [train\.py L242-L308](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L242-L308) supports flexible checkpoint loading:

| Parameter | Purpose | Effect |
| --- | --- | --- |
| \-\-load\_checkpoint\_path | Main checkpoint path | Loads model, optimizer, scheduler, step runner/train\.py255\-274 |
| \-\-load\_ema\_checkpoint\_path | EMA checkpoint path | Loads model weights for EMA initialization runner/train\.py276\-291 |
| \-\-load\_params\_only | Load only model weights | Skips optimizer/scheduler/step loading runner/train\.py250 |

### EMA Model Tracking

 The `EMAWrapper` class [ema\.py L21-L36](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/ema.py#L21-L36) maintains an exponential moving average of model parameters\. During evaluation [train\.py L360-L366](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L360-L366) the trainer evaluates both the training model and EMA model \(with suffix `ema{decay}_`\)\.

 Sources: [train\.py L242-L308](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/train.py#L242-L308) [ema\.py L21-L36](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/ema.py#L21-L36)

## Optimizer and Learning Rate Scheduling

### Optimizer Configuration

 The `get_optimizer()` function [training\.py L73-L115](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py#L73-L115) creates an AdamW optimizer with parameter grouping:

 **Parameter Grouping**: The `get_adamw()` function [training\.py L21-L71](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py#L21-L71) separates parameters:

| Group | Criteria | Weight Decay |
| --- | --- | --- |
| decay\_params | Tensors with ndim \>= 2 | Applied protenix/utils/training\.py47 |
| nodecay\_params | Tensors with ndim < 2 | Zero protenix/utils/training\.py48 |

### Learning Rate Scheduling

 The `get_lr_scheduler()` function [lr\_scheduler\.py L31-L48](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/lr_scheduler.py#L31-L48) supports multiple strategies, including `AlphaFold3LRScheduler` and `FinetuneLRScheduler` [lr\_scheduler\.py L130-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/lr_scheduler.py#L130-L165)

 Sources: [training\.py L21-L115](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/training.py#L21-L115) [lr\_scheduler\.py L31-L48](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/utils/lr_scheduler.py#L31-L48)

## Performance Optimization

 The system supports several kernel optimizations for training performance:

 - **Triangle Attention**: Supports `triattention`, `cuequivariance`, `deepspeed`, and `torch` [kernels\.md?plain=1 L10-L38](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/kernels.md?plain=1#L10-L38)
- **Triangle Multiplicative**: Supports `cuequivariance` and `torch` [kernels\.md?plain=1 L39-L51](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/kernels.md?plain=1#L39-L51)
- **Fast LayerNorm**: Used by default, modified from FastFold and Oneflow [kernels\.md?plain=1 L3-L9](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/kernels.md?plain=1#L3-L9)

 Sources: [kernels\.md?plain=1 L1-L52](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/kernels.md?plain=1#L1-L52) [train\_demo\.sh L15-L19](https://github.com/bytedance/Protenix/blob/c3bfc365/train_demo.sh#L15-L19)

## Child Pages

 For detailed information on specific training topics, see:

 - [Training Data Preparation](https://deepwiki.com/bytedance/Protenix/6.1-training-data-preparation) — Detail the process of preparing training datasets, including weighted sampling and constraint generation\.
- [Training Execution](https://deepwiki.com/bytedance/Protenix/6.2-training-execution) — Explain the training loop, loss computation, optimizer configuration, and learning rate scheduling\.
- [Fine\-tuning](https://deepwiki.com/bytedance/Protenix/6.3-fine-tuning) — Guide for fine\-tuning models with different crop sizes, confidence\-only training, and parameter selection\.

---
*Source: [https://deepwiki.com/bytedance/Protenix/6-training-system](https://deepwiki.com/bytedance/Protenix/6-training-system) on DeepWiki*