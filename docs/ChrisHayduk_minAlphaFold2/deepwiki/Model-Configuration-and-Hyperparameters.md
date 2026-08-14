# Model Configuration and Hyperparameters

> **Relevant source files**
> * [minalphafold/trainer.py](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py)

This page documents the configuration system used in minAlphaFold2. The project utilizes three primary dataclasses to manage model architecture, data processing, and training orchestration. These configurations allow the system to scale from a "tiny" pedagogical model to the full AlphaFold2 architecture.

## Configuration Architecture

The configuration is split into three distinct domains: `ModelConfig` (managed via `SimpleNamespace`), `DataConfig`, and `TrainingConfig`. These classes ensure that hyperparameters are centralized and easily adjustable for different experiment profiles.

### Data to Code Entity Mapping

The following diagram maps the high-level configuration concepts to their respective implementation classes and factory functions in the codebase.

**Configuration Entity Map**

```

```

Sources: [minalphafold/trainer.py L25-L68](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L25-L68)

 [minalphafold/trainer.py L71-L82](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L71-L82)

 [minalphafold/trainer.py L85-L107](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L85-L107)

---

## Model Configuration

The model configuration controls the dimensionality and depth of every sub-module in the AlphaFold2 architecture. While not a formal dataclass, it is managed as a `SimpleNamespace` created via the `_make_model_config` helper [minalphafold/trainer.py L25-L68](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L25-L68)

### Key Hyperparameter Fields

| Category | Field | Description |
| --- | --- | --- |
| **Dimensions** | `c_m`, `c_s`, `c_z` | Channels for MSA representation, Single representation, and Pair representation [minalphafold/trainer.py L27-L29](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L27-L29) |
| **Evoformer** | `num_evoformer` | Number of Evoformer blocks in the main stack [minalphafold/trainer.py L48](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L48-L48) |
| **Evoformer** | `evoformer_msa_dropout` | Dropout rate for MSA-related layers in Evoformer [minalphafold/trainer.py L49](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L49-L49) |
| **Evoformer** | `evoformer_pair_dropout` | Dropout rate for Pair-related layers in Evoformer [minalphafold/trainer.py L50](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L50-L50) |
| **Structure** | `structure_module_layers` | Number of iterative updates in the Structure Module [minalphafold/trainer.py L52](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L52-L52) |
| **IPA** | `ipa_num_heads` | Number of attention heads in Invariant Point Attention [minalphafold/trainer.py L55](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L55-L55) |
| **IPA** | `ipa_n_query_points` | Number of query points per head in IPA [minalphafold/trainer.py L57](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L57-L57) |
| **Templates** | `template_pair_num_blocks` | Depth of the Template Pair Stack [minalphafold/trainer.py L40](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L40-L40) |

### Built-in Model Profiles

minAlphaFold2 provides three standard profiles to facilitate different hardware constraints and research goals:

1. **`tiny` (default)**: A minimal profile with 1 Evoformer block and small dimensions (e.g., `c_m=32`, `c_z=16`), intended for CI/CD and unit testing [minalphafold/trainer.py L25-L68](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L25-L68)
2. **`medium`**: A pedagogical profile for local experiments. Increases Evoformer depth to 4 and expands dimensions (e.g., `c_m=128`, `c_z=64`) [minalphafold/trainer.py L113-L145](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L113-L145)
3. **`alphafold2`**: Matches the official AlphaFold2 monomer hyperparameters. Features 48 Evoformer blocks, 8 Structure Module layers, and full dimensionality (e.g., `c_m=256`, `c_z=128`) [minalphafold/trainer.py L148-L187](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L148-L187)

Sources: [minalphafold/trainer.py L109-L187](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L109-L187)

---

## Data Configuration

The `DataConfig` dataclass defines how protein features are sampled, cropped, and augmented before being fed into the model.

**Data Flow and Hyperparameters**

```

```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `crop_size` | `int` | 128 | Number of contiguous residues to sample from the sequence [minalphafold/trainer.py L76](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L76-L76) |
| `msa_depth` | `int` | 64 | Number of MSA sequences to include in the main MSA representation [minalphafold/trainer.py L77](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L77-L77) |
| `extra_msa_depth` | `int` | 128 | Number of extra MSA sequences for the Extra MSA Stack [minalphafold/trainer.py L78](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L78-L78) |
| `masked_msa_probability` | `float` | 0.15 | Probability of masking a token for the Masked MSA Head objective [minalphafold/trainer.py L81](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L81-L81) |
| `block_delete_training_msa` | `bool` | True | Whether to apply stochastic block deletion to MSAs during training [minalphafold/trainer.py L80](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L80-L80) |

Sources: [minalphafold/trainer.py L71-L82](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L71-L82)

---

## Training Configuration

The `TrainingConfig` dataclass manages the optimization loop, including learning rate schedules and the transition between training phases.

### Optimization and Learning Rate

The trainer supports a `warmup_cosine` schedule and constant learning rates.

* **Warmup**: Controlled by `warmup_steps` [minalphafold/trainer.py L96](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L96-L96)
* **Gradient Clipping**: Regulated by `grad_clip_norm`, defaulting to 1.0 [minalphafold/trainer.py L97](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L97-L97)
* **Recycling**: The number of recycling iterations is set via `n_cycles` [minalphafold/trainer.py L101](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L101-L101)

### Finetuning Logic

The system distinguishes between standard training and a "finetuning" phase.

* `finetune`: A boolean flag to enable finetuning-specific loss weights [minalphafold/trainer.py L103](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L103-L103)
* `finetune_start_step`: An optional integer defining the global step at which the model should switch to the finetuning configuration [minalphafold/trainer.py L104](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L104-L104)

### TrainingConfig Field Reference

| Field | Type | Default | Role |
| --- | --- | --- | --- |
| `learning_rate` | `float` | 1e-4 | Base learning rate for the Adam optimizer [minalphafold/trainer.py L89](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L89-L89) |
| `lr_schedule` | `str` | "constant" | Schedule type (e.g., "constant", "warmup_cosine") [minalphafold/trainer.py L95](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L95-L95) |
| `batch_size` | `int` | 1 | Number of samples per gradient update [minalphafold/trainer.py L88](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L88-L88) |
| `n_ensemble` | `int` | 1 | Number of ensemble iterations for inference [minalphafold/trainer.py L102](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L102-L102) |

Sources: [minalphafold/trainer.py L85-L107](https://github.com/ChrisHayduk/minAlphaFold2/blob/d0d066ad/minalphafold/trainer.py#L85-L107)