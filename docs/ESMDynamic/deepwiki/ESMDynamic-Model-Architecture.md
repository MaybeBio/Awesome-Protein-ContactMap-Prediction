# ESMDynamic Model Architecture

> **Relevant source files**
> * [Dockerfile](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/Dockerfile)
> * [esm/esmdynamic/dynamic_module.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py)
> * [esm/esmdynamic/esmdynamic.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py)
> * [esm/esmdynamic/predict.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/predict.py)
> * [esm/esmdynamic/pretrained.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/pretrained.py)
> * [esm/esmfold/v1/pretrained.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmfold/v1/pretrained.py)
> * [esm/esmfold/v1/tri_self_attn_block.py](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmfold/v1/tri_self_attn_block.py)

The `ESMDynamic` model is an extension of the ESMFold architecture designed to predict protein dynamics, including dynamic contact maps, contact frequency, and residue-pair kinetics from sequence information alone. It leverages the representations learned by a frozen ESM-2 language model and the geometric reasoning capabilities of the ESMFold trunk, while introducing specialized heads for dynamic properties.

## 1. High-Level Model Composition

The `ESMDynamic` class serves as the primary entry point. It wraps a pretrained `ESMFold` instance and appends three specialized `DynamicHead` modules to predict different aspects of protein motion.

### Natural Language to Code Entity Mapping: Model Entry

| System Concept | Code Entity | File Reference |
| --- | --- | --- |
| **Top-level Model** | `ESMDynamic` | [esm/esmdynamic/esmdynamic.py L202](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L202-L202) |
| **Prediction Head** | `DynamicHead` | [esm/esmdynamic/esmdynamic.py L32](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L32-L32) |
| **Folding Trunk Sub-module** | `DynamicModule` | [esm/esmdynamic/dynamic_module.py L35](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L35-L35) |
| **Recycling Configuration** | `DynamicModuleConfig` | [esm/esmdynamic/dynamic_module.py L17](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L17-L17) |

### Data Flow Overview

The model processes a sequence through the frozen `ESMFold` trunk to extract structural embeddings, which are then fused with task-specific transitions before entering the dynamic recycling loop.


**Sources:** [esm/esmdynamic/esmdynamic.py L202-L258](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L202-L258)

 [esm/esmdynamic/dynamic_module.py L35-L73](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L35-L73)

---

## 2. Feature Integration and Transitions

Before dynamic prediction, the model integrates static structural features from `ESMFold` into the dynamic heads. This is handled by `seq_transition` and `pair_transition` layers within each `DynamicHead` [esm/esmdynamic/esmdynamic.py L67-L76](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L67-L76)

* **Sequence Transition:** Concatenates `lddt_logits` (reshaped to include atom-level detail) and `lm_logits` from ESM-2, projecting them to the `seq_state_dim` [esm/esmdynamic/esmdynamic.py L121-L125](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L121-L125)
* **Pair Transition:** Concatenates `ptm_logits` and `distogram_logits` from ESMFold, projecting them to the `pair_state_dim` [esm/esmdynamic/esmdynamic.py L127-L130](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L127-L130)

These integrated features are added as residuals to the original `s_s` and `s_z` states from the folding trunk before being passed to the `DynamicModule` [esm/esmdynamic/esmdynamic.py L125-L130](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L125-L130)

**Sources:** [esm/esmdynamic/esmdynamic.py L67-L76](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L67-L76)

 [esm/esmdynamic/esmdynamic.py L121-L130](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L121-L130)

---

## 3. DynamicModule and Recycling Loop

The `DynamicModule` is a specialized version of the ESMFold `FoldingTrunk`. It implements a recycling loop that iteratively refines the sequence and pair representations specifically for dynamic tasks.

### Architecture of DynamicModule

* **Blocks:** Uses `TriangularSelfAttentionBlock` [esm/esmdynamic/dynamic_module.py L51-L64](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L51-L64)  to process pairwise interactions.
* **Recycling:** Implements a loop (default `max_recycles=4`) where the output of one pass is normalized and added to the input of the next [esm/esmdynamic/dynamic_module.py L112-L120](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L112-L120)
* **Positional Encoding:** Uses `RelativePosition` [esm/esmdynamic/dynamic_module.py L52](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L52-L52)  to maintain spatial awareness during the refinement process.


**Sources:** [esm/esmdynamic/dynamic_module.py L77-L122](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/dynamic_module.py#L77-L122)

 [esm/esmfold/v1/tri_self_attn_block.py L25-L171](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmfold/v1/tri_self_attn_block.py#L25-L171)

---

## 4. Task-Specific Prediction Heads

The `DynamicHead` supports four `task_type` modes, each determining the output dimensionality and post-processing logic [esm/esmdynamic/esmdynamic.py L83-L92](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L83-L92)

### Task Types and Outputs

| Task Type | Purpose | Output Shape | Symmetry Enforcement |
| --- | --- | --- | --- |
| `classification` | Dynamic contact probability | `[B, n_cond, L, L]` | `(p + p.T) / 2` |
| `kinetics` | On/Off rates for contacts | `[B, n_cond, 2, L, L, n_classes]` | `(p + p.T) / 2` across L, L |
| `regression` | Contact frequency/occupancy | `[B, n_cond, L, L]` | Residual addition |
| `multiclass` | Discrete dynamic states | `[B, n_cond, L, L, n_classes]` | Softmax + Symmetrize |

### Symmetry and Kinetics

For the `kinetics` task, the model predicts two rates (On and Off) across `n_conditions` (temperatures) and `n_classes` (time bins). The logits are reshaped to a canonical order: `[B, n_conditions, n_rates, L, L, n_classes]` [esm/esmdynamic/esmdynamic.py L144-L148](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L144-L148)

 Symmetry is enforced by averaging the probabilities of residue pair $(i, j)$ and $(j, i)$ [esm/esmdynamic/esmdynamic.py L154-L155](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L154-L155)

### Confidence and Residual Heads

* **Confidence Head:** For classification/kinetics, a per-residue, per-temperature confidence score is predicted using `seq_state_dim` [esm/esmdynamic/esmdynamic.py L97-L103](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L97-L103)
* **Residual Head:** For regression, a pairwise residual is added to the main prediction to refine the contact frequency map [esm/esmdynamic/esmdynamic.py L108-L115](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L108-L115)

**Sources:** [esm/esmdynamic/esmdynamic.py L81-L118](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L81-L118)

 [esm/esmdynamic/esmdynamic.py L144-L170](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/esmdynamic.py#L144-L170)

---

## 5. Pretrained Weights and Initialization

The model is typically initialized via `esm.esmdynamic.pretrained.esmdynamic()`, which loads weights from a remote repository [esm/esmdynamic/pretrained.py L32-L36](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/pretrained.py#L32-L36)

* **State Dict Loading:** The loader uses `strict=False` because the `ESMFold` backbone is usually kept frozen and may be loaded separately or shared [esm/esmdynamic/pretrained.py L27](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/pretrained.py#L27-L27)
* **Validation:** The loader verifies that essential dynamic head keys (e.g., `dynamic_module`, `kinetic_module`) are present in the state dict before proceeding [esm/esmdynamic/pretrained.py L19-L25](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/pretrained.py#L19-L25)

**Sources:** [esm/esmdynamic/pretrained.py L6-L36](https://github.com/MaybeBio/esmdynamic/blob/b3c7f04f/esm/esmdynamic/pretrained.py#L6-L36)