---
title: "Loss Functions and Optimization"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/5.3-loss-functions-and-optimization
---
# Loss Functions and Optimization

# Running Training

> **Relevant source files**
> - [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)
> - [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)
> - [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

 This document covers the practical execution of Boltz model training, including starting training runs, monitoring progress, checkpoint management, and troubleshooting\. For information about data preparation, see [Data Preparation](https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration)\. For details about training configurations, see [Training Configuration](https://deepwiki.com/jwohlwend/boltz/5.2-training-stages)\.

## Training Execution Overview

 The Boltz training system uses PyTorch Lightning as the training orchestration framework, with training runs executed through the main training script\. The system supports both single\-GPU development runs and multi\-GPU production training with checkpointing, monitoring, and distributed execution capabilities\.

## Training Script Structure

 The training execution flow involves several key components that work together to orchestrate the training process:

 **Training Execution Flow**

  Sources: [training\.md?plain=1 L98-L111](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L98-L111) [scripts/train/train\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py)

## Debug Mode Execution

 Before running full training, the system provides a debug mode that simplifies execution for development and testing:

 **Debug vs Production Execution**

  Debug mode execution disables distributed training features and runs everything in a single process, making it easier to debug issues with data loading, model initialization, or configuration problems\.

 Sources: [training\.md?plain=1 L100-L103](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L100-L103)

## Training Modes

 The Boltz system supports different training modes based on the configuration used:

| Training Mode | Config File | Purpose | Key Components |
| --- | --- | --- | --- |
| Structure | structure\.yaml | Basic structure prediction training | AtomDiffusion, DistogramModule, basic losses |
| Confidence | confidence\.yaml | Confidence estimation training | ConfidenceModule, pLDDT/PAE prediction |
| Full | full\.yaml | Complete Boltz\-2 training | All modules including AffinityModule, BFactorModule |

 The training script automatically determines the model architecture and loss functions based on the configuration file provided\.

 Sources: [training\.md?plain=1 L108-L111](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L108-L111)

## Checkpoint Management and Resume

 The training system provides robust checkpoint management with automatic saving and resume capabilities:

 **Checkpoint and Resume Workflow**

  The resume functionality is specified in the configuration file using the `resume` parameter, which should point to a valid checkpoint file path\.

 Sources: [training\.md?plain=1 L55](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L55-L55)

## Resource Management and Scaling

 The training system supports flexible resource configuration through the PyTorch Lightning trainer settings:

### GPU Configuration

 The number of GPUs is controlled through the `trainer.devices` parameter in the configuration:

### Memory Management

 Training memory usage is controlled through crop size parameters:

| Parameter | Recommended Values | Memory Impact |
| --- | --- | --- |
| max\_tokens | 256, 384, 512 | Controls sequence length |
| max\_atoms | 2304, 3456, 4608 | Controls structure size |

 Smaller values reduce memory usage but may impact training quality, while larger values require more GPU memory but can improve model performance\.

 Sources: [training\.md?plain=1 L64-L68](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L64-L68)

## Monitoring and Logging

 The training system integrates with Weights & Biases for experiment tracking and provides comprehensive logging:

 **Training Monitoring Components**

  In debug mode, Weights & Biases logging is disabled to simplify development workflows\.

 Sources: [training\.md?plain=1 L100-L103](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L100-L103)

## Troubleshooting Common Issues

### Data Loading Issues

 Common data loading problems include:

 - **Missing Data Files**: Ensure all dataset paths in the configuration point to valid directories
- **Insufficient Disk Space**: Training data requires ~250GB of storage
- **Permission Issues**: Verify read access to data directories and write access to output directories

### Memory Issues

 GPU memory problems can be addressed by:

 - Reducing `max_tokens` and `max_atoms` in the configuration
- Using gradient accumulation to simulate larger batch sizes
- Enabling mixed precision training if supported

### Configuration Errors

 Configuration validation errors typically occur due to:

 - Invalid file paths in dataset configurations
- Incompatible model and data parameters
- Missing required configuration sections

 The debug mode is particularly useful for identifying and resolving configuration issues before launching resource\-intensive production runs\.

 Sources: [training\.md?plain=1 L100-L106](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L100-L106)

## Example Training Commands

### Basic Structure Training

### Confidence Model Training

### Resume from Checkpoint

  Sources: [training\.md?plain=1 L102-L111](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L102-L111)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/5.3-loss-functions-and-optimization](https://deepwiki.com/jwohlwend/boltz/5.3-loss-functions-and-optimization) on DeepWiki*