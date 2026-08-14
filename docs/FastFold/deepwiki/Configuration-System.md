# Configuration System

> **Relevant source files**
> * [LICENSE](https://github.com/hpcaitech/FastFold/blob/eba49680/LICENSE)
> * [README.md](https://github.com/hpcaitech/FastFold/blob/eba49680/README.md?plain=1)
> * [environment.yml](https://github.com/hpcaitech/FastFold/blob/eba49680/environment.yml)
> * [fastfold/config.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py)
> * [fastfold/model/fastnn/kernel/layer_norm.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/model/fastnn/kernel/layer_norm.py)
> * [fastfold/relax/relax.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/relax.py)
> * [fastfold/relax/utils.py](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/utils.py)
> * [inference.py](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py)

FastFold's configuration system provides a structured, preset-based approach to defining model architectures, training parameters, data processing settings, and loss functions. It uses the `ml_collections` library to create hierarchical configuration dictionaries with shared parameter references that propagate changes across the entire configuration tree.

For information about using configurations during inference, see [Inference Pipeline](/hpcaitech/FastFold/5-inference-pipeline). For training-specific configuration usage, see [Training System](/hpcaitech/FastFold/7-training-system).

---

## System Architecture

The configuration system is built on three core components: **FieldReferences** for shared parameters, a **base configuration dictionary** with complete settings, and the **model_config() function** that applies preset-specific modifications.

**Configuration Generation Flow**

```

```

Sources: [fastfold/config.py L1-L607](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L1-L607)

---

## FieldReference System

FieldReferences enable centralized parameter management by creating shared references that automatically propagate across the configuration. When a FieldReference is updated, all locations that reference it receive the new value.

**FieldReference Propagation Mechanism**

```

```

**Defined FieldReferences**

| FieldReference | Default Value | Type | Purpose |
| --- | --- | --- | --- |
| `c_z` | 128 | int | Pair representation channel dimension |
| `c_m` | 256 | int | MSA representation channel dimension |
| `c_t` | 64 | int | Template embedding channel dimension |
| `c_e` | 64 | int | Extra MSA embedding channel dimension |
| `c_s` | 384 | int | Single representation channel dimension |
| `blocks_per_ckpt` | None | int | Gradient checkpointing frequency |
| `chunk_size` | None | int | Memory optimization chunk size |
| `aux_distogram_bins` | 64 | int | Auxiliary distogram prediction bins |
| `tm_enabled` | False | bool | Enable TM-score prediction head |
| `eps` | 1e-8 | float | Numerical stability epsilon |
| `templates_enabled` | True | bool | Enable template embeddings |
| `embed_template_torsion_angles` | True | bool | Embed template torsion angles |

Sources: [fastfold/config.py L128-L139](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L128-L139)

---

## Base Configuration Structure

The base configuration dictionary contains six major sections, each controlling a different aspect of the system.

```

```

Sources: [fastfold/config.py L146-L533](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L146-L533)

---

## Model Presets

The `model_config()` function supports 18 named presets that modify the base configuration for specific use cases. Each preset follows AlphaFold's published model variants.

**Preset Decision Tree**

```

```

**Model Preset Summary**

| Preset Name | Templates | Max Extra MSA | TM Head | Multimer | Notes |
| --- | --- | --- | --- | --- | --- |
| `initial_training` | ✓ | 1024 | ✗ | ✗ | Base configuration |
| `finetuning` | ✓ | 5120 | ✗ | ✗ | Larger MSA, violation loss enabled |
| `model_1` | ✓ | 5120 | ✗ | ✗ | AlphaFold Model 1.1.1 |
| `model_2` | ✓ | 1024 | ✗ | ✗ | AlphaFold Model 1.1.2 |
| `model_3` | ✗ | 5120 | ✗ | ✗ | AlphaFold Model 1.2.1 |
| `model_4` | ✗ | 5120 | ✗ | ✗ | AlphaFold Model 1.2.2 |
| `model_5` | ✗ | 1024 | ✗ | ✗ | AlphaFold Model 1.2.3 |
| `model_1_ptm` | ✓ | 5120 | ✓ | ✗ | Model 1 + pTM prediction |
| `model_2_ptm` | ✓ | 1024 | ✓ | ✗ | Model 2 + pTM prediction |
| `model_3_ptm` | ✗ | 5120 | ✓ | ✗ | Model 3 + pTM prediction |
| `model_4_ptm` | ✗ | 5120 | ✓ | ✗ | Model 4 + pTM prediction |
| `model_5_ptm` | ✗ | 1024 | ✓ | ✗ | Model 5 + pTM prediction |
| `model_X_multimer` | ✓ | varies | ✓ | ✓ | Multimer variant (any X) |

**Multimer-Specific Modifications**

When `"multimer"` appears in the model name, the following changes are applied in addition to base model settings:

```

```

Sources: [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

 [fastfold/config.py L535-L606](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L535-L606)

---

## Configuration Usage Patterns

**Basic Configuration Retrieval**

```

```

Sources: [inference.py L129](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L129-L129)

 [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

**Runtime Configuration Modification**

```

```

Sources: [inference.py L129-L145](https://github.com/hpcaitech/FastFold/blob/eba49680/inference.py#L129-L145)

**Accessing Configuration Values**

```

```

---

## Customization and Extension

**Creating Custom Presets**

To add a new model preset, extend the `model_config()` function:

```

```

**Defining New FieldReferences**

```

```

**Configuration Validation**

The configuration system performs basic validation through ml_collections:

* Type checking via `field_type` parameter in FieldReferences
* Immutability options via ConfigDict settings
* Nested structure enforcement

However, semantic validation (e.g., ensuring crop_size ≤ max sequence length) must be performed at runtime by the consuming code.

Sources: [fastfold/config.py L128-L139](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L128-L139)

 [fastfold/config.py L30-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L30-L125)

---

## Configuration Sections Reference

### Data Configuration

The `data` section controls all aspects of feature generation and data preprocessing.

**Common Settings** ([fastfold/config.py L149-L244](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L149-L244)

):

* `feat`: Dictionary of expected feature shapes with placeholders (NUM_RES, NUM_MSA_SEQ, etc.)
* `masked_msa`: MSA masking probabilities for training
* `max_extra_msa`: Maximum extra MSA sequences (1024 default, 5120 for some models)
* `max_recycling_iters`: Number of recycling iterations (3)
* `template_features`: List of template feature names to extract
* `unsupervised_features`: List of features available without ground truth
* `use_templates`: Whether to use template information
* `use_template_torsion_angles`: Whether to embed template torsions

**Mode-Specific Settings**:

* `predict` ([fastfold/config.py L255-L266](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L255-L266) ): Inference settings with no cropping
* `eval` ([fastfold/config.py L267-L278](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L267-L278) ): Evaluation with supervision
* `train` ([fastfold/config.py L279-L294](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L279-L294) ): Training with cropping, subsample templates

### Globals Configuration

The `globals` section ([fastfold/config.py L304-L314](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L304-L314)

) contains system-wide parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| `blocks_per_ckpt` | int/None | Gradient checkpointing frequency (None = no checkpointing) |
| `chunk_size` | int/None | Memory optimization chunk size (None = no chunking) |
| `c_z` | int | Pair representation channels (128) |
| `c_m` | int | MSA representation channels (256) |
| `c_t` | int | Template embedding channels (64) |
| `c_e` | int | Extra MSA embedding channels (64) |
| `c_s` | int | Single representation channels (384) |
| `eps` | float | Numerical stability epsilon (1e-8) |
| `is_multimer` | bool | Multimer mode flag (False) |

### Model Configuration

The `model` section ([fastfold/config.py L315-L460](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L315-L460)

) defines the neural network architecture:

* **input_embedder**: Target features and MSA embedding parameters
* **recycling_embedder**: Recycling embedding with distance bins
* **template**: Template processing with distogram, angle, and pair embedders
* **extra_msa**: Extra MSA stack configuration (4 blocks)
* **evoformer_stack**: Main Evoformer with 48 blocks, attention heads, dropout
* **structure_module**: IPA-based structure prediction (8 blocks, 12 heads)
* **heads**: Prediction heads (lddt, distogram, masked_msa, experimentally_resolved, tm)

### Loss Configuration

The `loss` section ([fastfold/config.py L468-L530](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L468-L530)

) defines training objectives and their weights:

| Loss Component | Default Weight | Purpose |
| --- | --- | --- |
| `distogram` | 0.3 | Distance distribution prediction |
| `fape` | 1.0 | Frame aligned point error (backbone + sidechain) |
| `lddt` | 0.01 | Local distance difference test |
| `masked_msa` | 2.0 | Masked MSA language modeling |
| `supervised_chi` | 1.0 | Sidechain torsion angle supervision |
| `violation` | 0.0 | Structural violation penalty (enabled for finetuning) |
| `tm` | 0.0 | TM-score prediction (enabled for PTM models) |
| `experimentally_resolved` | 0.0 | Experimental resolution prediction |

### Relax Configuration

The `relax` section ([fastfold/config.py L461-L467](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L461-L467)

) controls Amber relaxation:

* `max_iterations`: Maximum L-BFGS iterations (0 = no limit)
* `tolerance`: Energy tolerance in kcal/mol (2.39)
* `stiffness`: Restraint spring constant in kcal/mol Å² (10.0)
* `max_outer_iterations`: Violation-informed relax iterations (20)
* `exclude_residues`: Residues to exclude from restraints ([])

Sources: [fastfold/config.py L146-L533](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L146-L533)

 [fastfold/relax/relax.py L24-L92](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/relax/relax.py#L24-L92)

---

## Advanced Topics

### Training vs Inference Configurations

When `train=True` is passed to `model_config()`, specific changes are made:

```

```

This ensures gradient computation is memory-efficient during training while allowing full-speed inference with chunking.

### Low Precision Mode

When `low_prec=True` is specified, the configuration adapts for reduced precision:

```

```

The `set_inf()` function recursively replaces all `"inf"` fields in the configuration with the specified value to prevent numerical overflow in lower precision arithmetic.

### Configuration Inheritance

The system uses a deep copy pattern to ensure preset modifications don't affect the base configuration:

```

```

This allows multiple configurations to coexist without interference.

Sources: [fastfold/config.py L115-L125](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L115-L125)

 [fastfold/config.py L22-L28](https://github.com/hpcaitech/FastFold/blob/eba49680/fastfold/config.py#L22-L28)