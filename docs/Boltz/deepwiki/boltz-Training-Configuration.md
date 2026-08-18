---
title: "Training Configuration"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration
---
# Training Configuration

# Training Configuration

> **Relevant source files**
> - [examples/prot\_no\_msa\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/examples/prot_no_msa.yaml)
> - [pyproject\.toml](https://github.com/jwohlwend/boltz/blob/cb04aecc/pyproject.toml)
> - [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)
> - [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)
> - [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)
> - [src/boltz/model/layers/triangular\_mult\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/layers/triangular_mult.py)

 This document covers the configuration system for training Boltz models, including the structure of configuration files, different training modes, and key parameters for customizing the training process\. For information about preparing training data, see [Data Preparation](https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration)\. For details on actually running training jobs, see [Running Training](https://deepwiki.com/jwohlwend/boltz/5.3-loss-functions-and-optimization)\.

## Overview

 The Boltz training system uses YAML configuration files to specify all aspects of the training process, from data loading parameters to model architecture settings\. The system supports three distinct training modes, each with its own configuration template that can be customized for specific training objectives\.

## Configuration File Structure

 All training configurations follow a consistent hierarchical structure with four main sections:

  **Sources:** [structure\.yaml L1-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L1-L195) [confidence\.yaml L1-L202](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L1-L202) [full\.yaml L1-L201](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L1-L201)

### Trainer Configuration

 The `trainer` section configures PyTorch Lightning training parameters:

| Parameter | Description | Typical Values |
| --- | --- | --- |
| accelerator | Hardware accelerator type | gpu |
| devices | Number of devices to use | 1 |
| precision | Training precision | 32 |
| gradient\_clip\_val | Gradient clipping threshold | 10\.0 |
| max\_epochs | Maximum training epochs | \-1 \(infinite\) |
| accumulate\_grad\_batches | Gradient accumulation steps | 128 |

 **Sources:** [structure\.yaml L1-L7](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L1-L7)

### Data Configuration

 The data section defines datasets, processing pipelines, and resource limits:

  **Sources:** [structure\.yaml L22-L76](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L22-L76)

#### Dataset Configuration

 Each dataset is configured with a `DatasetConfig` object:

  **Sources:** [structure\.yaml L24-L34](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L24-L34)

### Model Configuration

 The model section specifies the target model class and all architecture parameters:

  **Sources:** [structure\.yaml L77-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L77-L195)

## Training Modes

 The Boltz system supports three distinct training modes, each with specific objectives and configurations:

### Structure Training Mode

 Structure\-only training focuses on learning accurate 3D molecular structure prediction using diffusion models\.

 **Key Configuration Parameters:**

 - `structure_prediction_training`: Not explicitly set \(defaults to `true`\)
- `confidence_prediction`: `false`
- Higher `sampling_steps` during training: `20`
- `diffusion_samples`: `2`

  **Sources:** [structure\.yaml L127-L148](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L127-L148)

### Confidence Training Mode

 Confidence\-only training learns to predict reliability metrics \(pLDDT, PAE, PDE\) from pre\-trained structure models\.

 **Key Configuration Parameters:**

 - `structure_prediction_training`: `false`
- `confidence_prediction`: `true`
- `pretrained`: Points to structure checkpoint
- `load_confidence_from_trunk`: `false` \(when resuming from confidence checkpoint\)

  **Sources:** [confidence\.yaml L129-L132](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L129-L132)

### Full Training Mode

 Full training combines both structure and confidence prediction in a unified training loop\.

 **Key Configuration Parameters:**

 - `structure_prediction_training`: `true`
- `confidence_prediction`: `true`
- `confidence_imitate_trunk`: `true`
- `run_confidence_sequentially`: `true` \(during validation\)

  **Sources:** [full\.yaml L128-L132](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L128-L132)

## Critical Configuration Parameters

### Data Limits and Resource Management

 These parameters control memory usage and computational requirements:

| Parameter | Structure Mode | Confidence Mode | Full Mode | Description |
| --- | --- | --- | --- | --- |
| max\_tokens | 512 | 512 | 512 | Maximum sequence length |
| max\_atoms | 4608 | 4608 | 4608 | Maximum number of atoms |
| max\_seqs | 2048 | 2048 | 2048 | Maximum MSA depth |
| batch\_size | 1 | 1 | 1 | Batch size per device |
| accumulate\_grad\_batches | 128 | 128 | 128 | Gradient accumulation |

 **Alternative Resource Settings:**

 - For smaller GPUs: `max_tokens: 256, max_atoms: 2304`
- For medium GPUs: `max_tokens: 384, max_atoms: 3456`

 **Sources:** [structure\.yaml L52-L59](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L52-L59) [training\.md?plain=1 L68](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L68-L68)

### Learning Rate and Optimization

 All training modes use the AlphaFold3\-style learning rate scheduler:

  **Sources:** [structure\.yaml L152-L158](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L152-L158)

### Loss Weights and Training Objectives

 Different training modes emphasize different loss components:

| Loss Component | Structure Mode | Confidence Mode | Full Mode |
| --- | --- | --- | --- |
| diffusion\_loss\_weight | 4\.0 | 4\.0 | 4\.0 |
| confidence\_loss\_weight | 1e\-4 | 3e\-3 | 3e\-3 |
| distogram\_loss\_weight | 3e\-2 | 3e\-2 | 3e\-2 |

 **Sources:** [structure\.yaml L146-L148](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L146-L148) [confidence\.yaml L150-L152](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L150-L152)

## Path Configuration

 All configuration files require setting specific data paths:

  **Required Path Updates:**

 - `output`: Directory for training outputs and checkpoints
- `target_dir`: Directory containing processed structure files
- `msa_dir`: Directory containing processed MSA files
- `symmetries`: Path to molecular symmetry information file
- `pretrained`: Path to pretrained checkpoint \(for confidence/full training\)

 **Sources:** [structure\.yaml L15-L34](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L15-L34) [training\.md?plain=1 L54-L63](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L54-L63)

## Advanced Configuration Options

### Multi\-Dataset Training

 The system supports training on multiple datasets with different sampling probabilities:

  **Sources:** [training\.md?plain=1 L74-L96](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L74-L96)

### Physical Guidance and Steering

 Optional steering parameters for physics\-based guidance during training:

  **Sources:** [structure\.yaml L188-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L188-L195)

### Activation Checkpointing and Memory Optimization

 Memory\-efficient training options for large models:

  **Sources:** [structure\.yaml L104-L112](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L104-L112)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration](https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration) on DeepWiki*