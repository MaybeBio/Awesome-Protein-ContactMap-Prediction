# Model Presets

> **Relevant source files**
> * [LICENSE](https://github.com/hpcaitech/FastFold/blob/eba49680/LICENSE)
> * [fastfold/config.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py)
> * [fastfold/model/fastnn/kernel/layer_norm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/layer_norm.py)
> * [fastfold/relax/relax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/relax.py)
> * [fastfold/relax/utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/utils.py)

This page documents the available model configuration presets in FastFold and their differences. Model presets provide pre-configured parameter sets that match specific AlphaFold model variants described in the AlphaFold2 supplementary materials.

For details on individual configuration parameters, see [Configuration Parameters](/hpcaitech/FastFold/3.2-configuration-parameters). For information on how configurations are used during training, see [Training System](/hpcaitech/FastFold/7-training-system).

---

## Overview

FastFold uses `ml_collections.ConfigDict` to manage model configurations. The `model_config()` function in [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

 provides named presets that modify a base configuration to create specific model variants. These presets correspond to the models described in the AlphaFold2 paper's supplementary tables 4 and 5.

### Configuration System Architecture

```

```

**Sources:** [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

 [fastfold/config.py L146-L533](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L146-L533)

---

## Using Model Presets

To instantiate a model configuration, call the `model_config()` function with the preset name:

```

```

The function accepts three parameters:

* `name`: Preset name (string)
* `train`: Whether to configure for training (bool, default=False)
* `low_prec`: Whether to use low precision settings (bool, default=False)

**Sources:** [fastfold/config.py L30](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L30)

---

## Monomer Model Presets

The five monomer model variants correspond to AlphaFold2 Supplementary Table 5. The key differences are in template usage and MSA depth.

### Model Variant Comparison

| Preset | Templates Enabled | Template Torsions | max_extra_msa | reduce_by_templates | Reference |
| --- | --- | --- | --- | --- | --- |
| `model_1` | ✓ | ✓ | 5120 | ✓ | Model 1.1.1 |
| `model_2` | ✓ | ✓ | 1024 | ✓ | Model 1.1.2 |
| `model_3` | ✗ | ✗ | 5120 | ✗ | Model 1.2.1 |
| `model_4` | ✗ | ✗ | 5120 | ✗ | Model 1.2.2 |
| `model_5` | ✗ | ✗ | 1024 | ✗ | Model 1.2.3 |

**Sources:** [fastfold/config.py L41-L64](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L41-L64)

### Configuration Details by Model

```

```

**Sources:** [fastfold/config.py L41-L64](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L41-L64)

#### model_1

AlphaFold2 Model 1.1.1 - Full template support with maximum MSA depth:

* `c.data.common.max_extra_msa = 5120`
* `c.data.common.reduce_max_clusters_by_max_templates = True`
* `c.data.common.use_templates = True`
* `c.data.common.use_template_torsion_angles = True`
* `c.model.template.enabled = True`

**Sources:** [fastfold/config.py L41-L47](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L41-L47)

#### model_2

AlphaFold2 Model 1.1.2 - Template support with standard MSA depth:

* Uses default `max_extra_msa = 1024`
* `c.data.common.reduce_max_clusters_by_max_templates = True`
* `c.data.common.use_templates = True`
* `c.data.common.use_template_torsion_angles = True`
* `c.model.template.enabled = True`

**Sources:** [fastfold/config.py L48-L53](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L48-L53)

#### model_3

AlphaFold2 Model 1.2.1 - No templates, maximum MSA:

* `c.data.common.max_extra_msa = 5120`
* `c.model.template.enabled = False`

**Sources:** [fastfold/config.py L54-L57](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L54-L57)

#### model_4

AlphaFold2 Model 1.2.2 - No templates, maximum MSA:

* `c.data.common.max_extra_msa = 5120`
* `c.model.template.enabled = False`

**Sources:** [fastfold/config.py L58-L61](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L58-L61)

#### model_5

AlphaFold2 Model 1.2.3 - No templates, standard MSA:

* Uses default `max_extra_msa = 1024`
* `c.model.template.enabled = False`

**Sources:** [fastfold/config.py L62-L64](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L62-L64)

---

## PTM Model Presets

PTM (Predicted Template Modeling) variants add a TM-score prediction head to the corresponding base models. These models predict both structure and a confidence metric based on the TM-score.

### PTM Configuration Pattern

All PTM models follow the same pattern:

1. Inherit all settings from the corresponding base model
2. Enable the TM prediction head: `c.model.heads.tm.enabled = True`
3. Add TM loss weight: `c.loss.tm.weight = 0.1`

```

```

**Sources:** [fastfold/config.py L65-L93](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L65-L93)

### Available PTM Presets

| Preset | Base Model | Templates | max_extra_msa | TM Head | TM Loss Weight |
| --- | --- | --- | --- | --- | --- |
| `model_1_ptm` | model_1 | ✓ | 5120 | ✓ | 0.1 |
| `model_2_ptm` | model_2 | ✓ | 1024 | ✓ | 0.1 |
| `model_3_ptm` | model_3 | ✗ | 5120 | ✓ | 0.1 |
| `model_4_ptm` | model_4 | ✗ | 5120 | ✓ | 0.1 |
| `model_5_ptm` | model_5 | ✗ | 1024 | ✓ | 0.1 |

**Sources:** [fastfold/config.py L65-L93](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L65-L93)

---

## Training Presets

Training presets configure data processing and loss settings for different training phases.

### initial_training

Default training configuration matching AlphaFold2 Supplementary Table 4 "initial training" setting:

* Uses base configuration without modifications
* `max_extra_msa = 1024` (default)
* `crop_size = 256` (default)
* `max_msa_clusters = 128` (default)
* `violation.weight = 0.0` (default)

**Sources:** [fastfold/config.py L32-L34](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L32-L34)

### finetuning

Finetuning configuration matching AlphaFold2 Supplementary Table 4 "finetuning" setting:

* `c.data.common.max_extra_msa = 5120` (increased from 1024)
* `c.data.train.crop_size = 384` (increased from 256)
* `c.data.train.max_msa_clusters = 512` (increased from 128)
* `c.loss.violation.weight = 1.0` (enabled structure violation loss)

**Sources:** [fastfold/config.py L35-L40](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L35-L40)

### Training vs Inference Differences

When `train=True` is passed to `model_config()`:

* `c.globals.blocks_per_ckpt = 1` (enables gradient checkpointing)
* `c.globals.chunk_size = None` (disables chunking for full backpropagation)

**Sources:** [fastfold/config.py L115-L117](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L115-L117)

---

## Multimer Presets

Multimer presets configure the model for protein complex prediction. Any preset name containing "multimer" triggers multimer-specific settings.

### Multimer Configuration Updates

```

```

**Sources:** [fastfold/config.py L96-L111](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L96-L111)

### Key Multimer Changes

The multimer configuration applies these changes:

1. **Global Settings** * `c.globals.is_multimer = True`
2. **Data Configuration** * `c.data.predict.max_msa_clusters = 252` (increased from 128)
3. **Structure Module** * `c.model.structure_module.trans_scale_factor = 20` (increased from 10)
4. **Model Architecture Updates** * Applied via `multimer_model_config_update` dictionary
5. **Unsupervised Features** * Adds: `msa_mask`, `seq_mask`, `asym_id`, `entity_id`, `sym_id`

**Sources:** [fastfold/config.py L97-L111](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L97-L111)

### Multimer Model Config Updates

The `multimer_model_config_update` dictionary at [fastfold/config.py L535-L606](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L535-L606)

 contains multimer-specific model architecture changes:

#### Input Embedder Modifications

```

```

**Sources:** [fastfold/config.py L536-L545](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L536-L545)

#### Template Embedder Modifications

```

```

**Sources:** [fastfold/config.py L546-L581](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L546-L581)

#### Heads Modifications

```

```

**Sources:** [fastfold/config.py L582-L605](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L582-L605)

---

## Special Presets

### relax

The "relax" preset uses the base configuration without modifications. It is used for Amber relaxation configuration rather than model configuration.

**Sources:** [fastfold/config.py L94-L95](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L94-L95)

---

## Low Precision Mode

When `low_prec=True` is passed to `model_config()`, the following adjustments are made for numerical stability:

* `c.globals.eps = 1e-4` (increased from 1e-8)
* All `inf` values in the config are set to `1e4` (reduced from 1e8/1e9)

This mode helps with half-precision (FP16) training and inference where extreme values can cause numerical issues.

**Sources:** [fastfold/config.py L119-L123](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L119-L123)

---

## Field References

FastFold uses `ml_collections.FieldReference` to create linked configuration values that can be updated globally. Key field references defined at [fastfold/config.py L128-L139](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L128-L139)

:

| Field Reference | Default Value | Description |
| --- | --- | --- |
| `c_z` | 128 | Pair representation dimension |
| `c_m` | 256 | MSA representation dimension |
| `c_t` | 64 | Template pair dimension |
| `c_e` | 64 | Extra MSA dimension |
| `c_s` | 384 | Single representation dimension |
| `blocks_per_ckpt` | None | Blocks per gradient checkpoint |
| `chunk_size` | None | Chunking size for memory optimization |
| `aux_distogram_bins` | 64 | Distogram bins for auxiliary heads |
| `tm_enabled` | False | TM-score head enabled |
| `eps` | 1e-8 | Epsilon for numerical stability |
| `templates_enabled` | True | Template processing enabled |
| `embed_template_torsion_angles` | True | Embed template torsion angles |

These field references are used throughout the configuration to maintain consistency. Changing a field reference value updates all locations where it's referenced.

**Sources:** [fastfold/config.py L128-L139](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L128-L139)

---

## Preset Selection Guidelines

### For Inference

* **High accuracy (templates available)**: Use `model_1_ptm` or `model_2_ptm`
* **Fast inference (no templates)**: Use `model_3_ptm`, `model_4_ptm`, or `model_5_ptm`
* **Protein complexes**: Use multimer variants (e.g., "multimer" in preset name)

### For Training

* **Initial training**: Use `initial_training`
* **Finetuning**: Use `finetuning` with `train=True`
* **Low precision hardware**: Add `low_prec=True`

### Template Usage Considerations

Models with templates enabled require:

* Template search results (from `hhsearch` against PDB70)
* Additional computation time for template processing
* More memory for template features

Models without templates:

* Faster inference
* Lower memory requirements
* Rely solely on MSA co-evolution signals

**Sources:** [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)