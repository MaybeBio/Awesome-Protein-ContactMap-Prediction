# Quick Start: Training

> **Relevant source files**
> * [LICENSE](https://github.com/hpcaitech/FastFold/blob/eba49680/LICENSE)
> * [fastfold/config.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py)
> * [fastfold/data/data_modules.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py)
> * [fastfold/model/fastnn/kernel/layer_norm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/layer_norm.py)
> * [fastfold/relax/relax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/relax.py)
> * [fastfold/relax/utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/utils.py)
> * [fastfold/utils/tensor_utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/tensor_utils.py)
> * [fastfold/utils/test_utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/utils/test_utils.py)
> * [tests/test_train.py](https://github.com/hpcaitech/FastFold/blob/eba49680/tests/test_train.py)
> * [train.py](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py)

This guide demonstrates how to set up and run training experiments with FastFold's `train.py` script. It covers basic training invocation, dataset preparation, configuration options, and distributed execution. For comprehensive details about the training system architecture, loss functions, and advanced topics, see [Training System](/hpcaitech/FastFold/7-training-system).

For running inference on pretrained models, see [Quick Start: Inference](/hpcaitech/FastFold/2.2-quick-start:-inference). For installation and environment setup, see [Installation](/hpcaitech/FastFold/2.1-installation).

---

## Overview

FastFold training uses the `train.py` script to train AlphaFold models from scratch or fine-tune existing weights. The training pipeline integrates:

* **Data Loading**: OpenFold-compatible datasets with stochastic filtering
* **Model Architecture**: AlphaFold with FastNN optimizations via `inject_fastnn`
* **Distributed Training**: ColossalAI engine with optional Dynamic Axial Parallelism (DAP)
* **Optimization**: HybridAdam optimizer with custom learning rate scheduling

---

## Training Execution Flow

```

```

**Sources:** [train.py L36-L256](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L36-L256)

---

## Basic Training Command

### Minimal Example

```

```

### With Distillation Data

```

```

**Sources:** [train.py L36-L159](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L36-L159)

---

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--template_mmcif_dir` | str | Directory containing template mmCIF files for structural templates |
| `--max_template_date` | str | Cutoff date for templates (YYYY-MM-DD format). Training filters by target release date |
| `--train_data_dir` | str | Directory with training mmCIF files |
| `--train_alignment_dir` | str | Directory with precomputed training alignments (MSAs, templates) |
| `--train_chain_data_cache_path` | str | JSON cache mapping chain IDs to metadata (resolution, sequence, cluster size) |

### Data Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--distillation_data_dir` | str | None | Directory with PDB files for self-distillation |
| `--distillation_alignment_dir` | str | None | Precomputed alignments for distillation set |
| `--distillation_chain_data_cache_path` | str | None | Chain metadata cache for distillation |
| `--val_data_dir` | str | None | Validation mmCIF directory |
| `--val_alignment_dir` | str | None | Validation alignment directory |
| `--kalign_binary_path` | str | /usr/bin/kalign | Path to kalign binary for template realignment |

### Training Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--config_preset` | str | initial_training | Model config: `initial_training`, `finetuning`, `model_1-5`, `model_X_ptm` |
| `--train_epoch_len` | int | 10000 | Virtual epoch length (affects checkpoint/validation frequency) |
| `--max_epochs` | int | 10000 | Maximum training epochs |
| `--seed` | int | 42 | Random seed for reproducibility |

### Distributed Training

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--from_torch` | flag | False | Use `colossalai.launch_from_torch` instead of config file |
| `--dap_size` | int | 1 | Dynamic Axial Parallelism size (1 to nproc_per_node) |

### Logging and Checkpointing

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--log_path` | str | train_log | Directory for training logs |
| `--log_interval` | int | 1 | Log metrics every N steps |
| `--save_ckpt_path` | str | None | Checkpoint save directory (None = no saving) |
| `--save_ckpt_interval` | int | 1 | Save checkpoint every N epochs |

**Sources:** [train.py L37-L157](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L37-L157)

---

## Configuration Presets

The `--config_preset` argument selects predefined model configurations. Each preset modifies the base configuration with specific settings for data processing, model architecture, and loss weights.

### Available Presets

| Preset | Description | Key Settings |
| --- | --- | --- |
| `initial_training` | AlphaFold Suppl. Table 4, initial training phase | Default settings, `max_extra_msa=1024` |
| `finetuning` | AlphaFold Suppl. Table 4, finetuning phase | `max_extra_msa=5120`, `crop_size=384`, `max_msa_clusters=512`, violation loss weight=1.0 |
| `model_1` | Model 1.1.1 (with templates) | `max_extra_msa=5120`, templates enabled, template torsion angles enabled |
| `model_2` | Model 1.1.2 (with templates) | Templates enabled, template torsion angles enabled |
| `model_3` | Model 1.2.1 (no templates) | `max_extra_msa=5120`, templates disabled |
| `model_4` | Model 1.2.2 (no templates) | `max_extra_msa=5120`, templates disabled |
| `model_5` | Model 1.2.3 (no templates) | Templates disabled |
| `model_1_ptm` through `model_5_ptm` | PTM (predicted TM-score) variants | Same as base models + TM head enabled, TM loss weight=0.1 |

### Training-Specific Config Modifications

When `train=True` is passed to `model_config()`:

* `globals.blocks_per_ckpt = 1` - Enable gradient checkpointing per block
* `globals.chunk_size = None` - Disable chunking (train on full sequences)

**Sources:** [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

---

## Dataset Preparation

### Chain Data Cache Format

The chain data cache JSON files map chain IDs to metadata for stochastic filtering:

```

```

### Stochastic Filtering

The training pipeline applies two types of filters to control data distribution:

**Deterministic Filters** (hard filters):

* Resolution must be ≤ 9.0 Å
* No single amino acid can comprise > 80% of sequence

**Stochastic Filters** (probability-based):

* Cluster size filter: P = 1 / cluster_size
* Length filter: P = (1/512) × max(min(length, 512), 256)
* Combined probability: P_total = P_cluster × P_length

**Sources:** [fastfold/data/data_modules.py L225-L266](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L225-L266)

---

## Training Data Flow

```

```

**Sources:** [train.py L177-L251](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L177-L251)

 [fastfold/data/data_modules.py L479-L640](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L479-L640)

---

## ColossalAI Integration

### Initialization Options

#### Option 1: Launch from Torch (Recommended)

```

```

#### Option 2: Launch from Config File

```

```

### Engine Wrapping

The `colossalai.initialize()` call wraps all training components:

```

```

The resulting `engine` provides unified methods:

* `engine(batch)` - Forward pass
* `engine.criterion(output, batch)` - Loss computation
* `engine.backward(loss)` - Backward pass with gradient communication
* `engine.step()` - Optimizer step
* `engine.zero_grad()` - Zero gradients

**Sources:** [train.py L164-L220](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L164-L220)

---

## Dynamic Axial Parallelism (DAP)

DAP enables training on sequences longer than what fits in a single GPU by sharding along the residue axis. The `--dap_size` argument controls the parallelism degree.

### DAP Size Guidelines

| DAP Size | Use Case | Sequence Length Support | GPUs Required |
| --- | --- | --- | --- |
| 1 | Standard training | Up to ~1024 residues | 1 |
| 2 | Long sequences | Up to ~2048 residues | 2 |
| 4 | Very long sequences | Up to ~4096 residues | 4 |
| 8 | Ultra-long sequences | Up to ~8192 residues | 8 |

### Configuration

The DAP size is passed to ColossalAI's tensor parallel configuration:

```

```

During model execution, FastFold's distributed primitives (scatter, gather, all-to-all) handle cross-GPU communication automatically. For details, see [Dynamic Axial Parallelism](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap)).

**Sources:** [train.py L155-L166](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L155-L166)

---

## Loss Computation and Metrics

### Loss Breakdown Logging

The training loop computes and logs individual loss components:

```

```

Loss components logged:

* **FAPE**: Frame Aligned Point Error (backbone + sidechain)
* **distogram**: Predicted distance distribution
* **masked_msa**: Masked MSA prediction
* **lddt**: Local Distance Difference Test
* **supervised_chi**: Side-chain torsion angles
* **violation**: Structural violations (if weight > 0)
* **tm**: TM-score prediction (if PTM head enabled)

### Validation Metrics

Additional metrics computed during validation via `compute_validation_metrics()`:

* **RMSD**: Root mean square deviation from ground truth
* **GDT-TS**: Global Distance Test - Total Score
* **GDT-HA**: Global Distance Test - High Accuracy
* **TMScore**: Template Modeling score

**Sources:** [train.py L21-L33](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L21-L33)

 [train.py L226-L251](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L226-L251)

---

## Checkpointing

### Saving Checkpoints

Checkpoints are saved at intervals controlled by `--save_ckpt_interval`:

```

```

The saved checkpoint includes:

* Model weights (after ColossalAI wrapping)
* FastNN optimizations (inject_fastnn is part of model structure)

### Loading Checkpoints

To resume training from a checkpoint:

```

```

**Sources:** [train.py L253-L254](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L253-L254)

---

## Training Loop Structure

### Main Training Loop

```

```

### Recycling Selection

The model outputs contain predictions for each recycling iteration. Training uses only the final iteration:

```

```

This extracts the last slice along the recycling dimension for all tensors in the batch.

**Sources:** [train.py L224-L254](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L224-L254)

---

## Example Training Workflow

### 1. Prepare Data

```

```

### 2. Start Training

```

```

### 3. Monitor Training

Training logs contain:

```yaml
Training, Epoch: 0, Step: 100, Global_Step: 100, 
Loss: fape=2.145 distogram=1.823 masked_msa=0.756 lddt=0.234 ...
```

### 4. Distributed Training (Multi-GPU)

```

```

**Sources:** [train.py L1-L259](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L1-L259)

---

## Common Issues and Solutions

### Out of Memory

**Problem**: CUDA out of memory during training

**Solutions**:

1. Reduce crop size in config: `config.data.train.crop_size = 128`
2. Increase DAP size to shard across more GPUs
3. Reduce number of recycling iterations: `config.common.max_recycling_iters = 2`

### Slow Data Loading

**Problem**: Training bottlenecked by data loading

**Solutions**:

1. Increase `num_workers` in config: `config.data_module.data_loaders.num_workers = 32`
2. Use alignment index for faster lookup: `--_alignment_index_path`
3. Precompute and cache all features

### Validation Takes Too Long

**Problem**: Validation loop significantly slower than training

**Solution**: Validation runs on full-length sequences without cropping. Consider:

1. Reducing validation set size
2. Validating less frequently (every N epochs)
3. Using a separate validation script

**Sources:** [fastfold/config.py L279-L301](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L279-L301)

 [fastfold/data/data_modules.py L592-L639](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L592-L639)

---

## Next Steps

* **Advanced Training Configuration**: See [Training System](/hpcaitech/FastFold/7-training-system) for detailed architecture
* **Loss Function Components**: See [Loss Functions and Metrics](/hpcaitech/FastFold/7.3-loss-functions-and-metrics)
* **Data Pipeline Details**: See [Training Data Loading](/hpcaitech/FastFold/7.1-training-data-loading)
* **Distributed Strategies**: See [Dynamic Axial Parallelism](/hpcaitech/FastFold/8.1-dynamic-axial-parallelism-(dap))
* **Performance Tuning**: See [Performance Tuning Guide](/hpcaitech/FastFold/12-performance-tuning-guide)

**Sources:** [train.py L1-L259](https://github.com/hpcaitech/FastFold/blob/eba49680/train.py#L1-L259)

 [fastfold/config.py L1-L607](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L1-L607)

 [fastfold/data/data_modules.py L1-L640](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/data/data_modules.py#L1-L640)