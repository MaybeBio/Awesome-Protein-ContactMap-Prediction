# Model Execution

> **Relevant source files**
> * [alphafold/model/features.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/features.py)
> * [alphafold/model/model.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py)
> * [alphafold/model/modules.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py)
> * [alphafold/model/modules_multimer.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py)
> * [alphafold/model/tf/proteins_dataset.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/tf/proteins_dataset.py)

## Purpose and Scope

This document describes how AlphaFold models are executed at runtime. It covers the `RunModel` class, which serves as the primary interface for model inference, including parameter initialization, feature processing, Haiku functional transformations, JAX JIT compilation, and the prediction workflow that produces structural predictions and confidence metrics.

For information about the model architecture that gets executed, see [Evoformer Stack](/google-deepmind/alphafold/4.2-evoformer-stack) and [Structure Module](/google-deepmind/alphafold/4.3-structure-module). For configuration options that control execution, see [Configuration System](/google-deepmind/alphafold/4.1-configuration-system). For post-prediction structure refinement, see [Structure Relaxation](/google-deepmind/alphafold/6.2-structure-relaxation).

## RunModel Class Overview

The `RunModel` class ([alphafold/model/model.py L72-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L72-L184)

) is a container that wraps the AlphaFold neural network model for inference. It handles the transformation of the model into a functional form using Haiku, JIT compilation with JAX, parameter management, and the execution pipeline.

### Class Structure

```mermaid
flowchart TD

Constructor["RunModel.init[alphafold/model/model.py:75-102]<br>(config, params)"]
SetConfig["self.config = config"]
SetParams["self.params = params"]
CheckMode["Check multimer_mode<br>from config.model.global_config"]
MultimerFn["Define _forward_fn<br>modules_multimer.AlphaFold[alphafold/model/modules_multimer.py]"]
MonomerFn["Define _forward_fn<br>modules.AlphaFold[alphafold/model/modules.py]"]
Transform1["hk.transform(_forward_fn)"]
Transform2["hk.transform(_forward_fn)"]
JIT1["jax.jit(transform.apply)"]
JIT2["jax.jit(transform.init)"]
JIT3["jax.jit(transform.apply)"]
JIT4["jax.jit(transform.init)"]
Apply["self.apply[alphafold/model/model.py:101]"]
Init["self.init[alphafold/model/model.py:102]"]
Predict["predict(feat, random_seed)[alphafold/model/model.py:153]"]
InitParams["init_params(feat, random_seed)[alphafold/model/model.py:104]"]
Process["process_features(raw_features, random_seed)[alphafold/model/model.py:122]"]
EvalShape["eval_shape(feat)[alphafold/model/model.py:143]"]

Apply --> Predict
Init --> InitParams

subgraph RunModel_Methods ["RunModel_Methods"]
    Predict
    InitParams
    Process
    EvalShape
end

subgraph RunModel_Initialization ["RunModel_Initialization"]
    Constructor
    SetConfig
    SetParams
    CheckMode
    MultimerFn
    MonomerFn
    Transform1
    Transform2
    JIT1
    JIT2
    JIT3
    JIT4
    Apply
    Init
    Constructor --> SetConfig
    Constructor --> SetParams
    Constructor --> CheckMode
    CheckMode --> MultimerFn
    CheckMode --> MonomerFn
    MultimerFn --> Transform1
    MonomerFn --> Transform2
    Transform1 --> JIT1
    Transform1 --> JIT2
    Transform2 --> JIT3
    Transform2 --> JIT4
    JIT1 --> Apply
    JIT2 --> Init
    JIT3 --> Apply
    JIT4 --> Init
end
```

**Sources:** [alphafold/model/model.py L72-L103](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L72-L103)

 [alphafold/model/modules.py L134-L143](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L134-L143)

 [alphafold/model/modules_multimer.py L15-L21](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L15-L21)

### Key Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `config` | `ml_collections.ConfigDict` | Model configuration dictionary |
| `params` | `Mapping[str, Mapping[str, jax.Array]]` | Model parameters (weights) |
| `multimer_mode` | `bool` | Whether to use multimer architecture |
| `apply` | JIT-compiled function | Forward pass function for inference |
| `init` | JIT-compiled function | Parameter initialization function |

**Sources:** [alphafold/model/model.py L75-L102](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L75-L102)

## Haiku Transforms and JAX JIT Compilation

AlphaFold uses Haiku's functional programming paradigm combined with JAX's JIT compilation for efficient execution.

### Haiku Transform Process

```mermaid
flowchart TD

ForwardDef["def _forward_fn(batch):<br>model = modules.AlphaFold(config)<br>return model(batch, ...)"]
Transform["hk.transform(_forward_fn)"]
TransObj["Transformed Object<br>.apply(params, rng, batch)<br>.init(rng, batch)"]
JITApply["jax.jit(transform.apply)[alphafold/model/model.py:101]"]
JITInit["jax.jit(transform.init)[alphafold/model/model.py:102]"]
CompiledApply["Compiled Apply Function<br>XLA-optimized"]
CompiledInit["Compiled Init Function<br>XLA-optimized"]

ForwardDef --> Transform
TransObj --> JITApply
TransObj --> JITInit

subgraph Step_3_JAX_JIT_Compilation ["Step_3_JAX_JIT_Compilation"]
    JITApply
    JITInit
    CompiledApply
    CompiledInit
    JITApply --> CompiledApply
    JITInit --> CompiledInit
end

subgraph Step_2_Haiku_Transform ["Step_2_Haiku_Transform"]
    Transform
    TransObj
    Transform --> TransObj
end

subgraph Step_1_Define_Forward_Function ["Step_1_Define_Forward_Function"]
    ForwardDef
end
```

**Sources:** [alphafold/model/model.py L86-L102](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L86-L102)

### Forward Function Definitions

The forward function differs between monomer and multimer modes:

**Monomer Mode** ([alphafold/model/model.py L92-L99](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L92-L99)

):

```python
def _forward_fn(batch):    model = modules.AlphaFold(self.config.model)    return model(        batch,        is_training=False,        compute_loss=False,        ensemble_representations=True,    )
```

**Multimer Mode** ([alphafold/model/model.py L86-L88](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L86-L88)

):

```python
def _forward_fn(batch):    model = modules_multimer.AlphaFold(self.config.model)    return model(batch, is_training=False)
```

**Sources:** [alphafold/model/model.py L84-L99](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L84-L99)

### JIT Compilation Benefits

JAX's JIT compilation ([alphafold/model/model.py L101-L102](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L101-L102)

) provides:

* XLA-optimized execution on GPU/TPU
* Automatic parallelization
* Efficient memory usage
* Compiled once, executed many times

## Prediction Workflow

The complete prediction workflow involves several stages, coordinated by methods of the `RunModel` class.

```mermaid
flowchart TD

Start["Input: raw_features<br>(NumPy dict from data pipeline)"]
Process["process_features()[alphafold/model/model.py:122]<br>Prepare features for model"]
MultFeat["Return raw_features<br>(no processing needed)"]
MonoFeat["np_example_to_features()[alphafold/model/features.py:45]<br>TensorFlow pipeline<br>Cropping, MSA sampling, etc."]
CheckParams["Parameters<br>initialized?"]
InitParams["init_params()[alphafold/model/model.py:104]<br>Initialize randomly or<br>load from checkpoint"]
Predict["predict(feat, random_seed)[alphafold/model/model.py:153]<br>Run model inference"]
Apply["self.apply(params, rng, feat)<br>JIT-compiled forward pass"]
Block["Block until ready<br>tree.map(block_until_ready)[alphafold/model/model.py:179]"]
Confidence["get_confidence_metrics()[alphafold/model/model.py:31]<br>Compute pLDDT, pTM, ipTM, PAE"]
Result["Return: prediction_result<br>+ confidence_metrics"]

Start --> Process
Process --> MultFeat
Process --> MonoFeat
MultFeat --> CheckParams
MonoFeat --> CheckParams
CheckParams --> InitParams
CheckParams --> Predict
InitParams --> Predict
Predict --> Apply
Apply --> Block
Block --> Confidence
Confidence --> Result
```

**Sources:** [alphafold/model/model.py L104-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L104-L184)

 [alphafold/model/features.py L45-L76](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/features.py#L45-L76)

### Parameter Initialization

The `init_params` method ([alphafold/model/model.py L104-L120](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L104-L120)

) handles model parameter initialization:

```mermaid
flowchart TD

Entry["init_params(feat, random_seed=0)"]
Check["self.params<br>exists?"]
Skip["Skip initialization<br>Use provided params"]
Random["Generate random seed<br>jax.random.PRNGKey(random_seed)"]
CallInit["self.init(rng, feat)<br>Initialize parameters"]
Convert["hk.data_structures.to_mutable_dict()[alphafold/model/model.py:119]<br>Convert to mutable dict"]
Store["self.params = initialized params"]
Warn["Log warning:<br>'Initialized parameters randomly'"]

Entry --> Check
Check --> Skip
Check --> Random
Random --> CallInit
CallInit --> Convert
Convert --> Store
Store --> Warn
```

**Sources:** [alphafold/model/model.py L104-L120](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L104-L120)

### Feature Processing

Feature processing differs significantly between monomer and multimer modes:

| Mode | Processing | Implementation |
| --- | --- | --- |
| **Multimer** | No processing | Returns `raw_features` as-is ([alphafold/model/model.py L137](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L137-L137) <br> ) |
| **Monomer** | TensorFlow pipeline | Calls `features.np_example_to_features()` ([alphafold/model/model.py L139-L141](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L139-L141) <br> ) |

**Sources:** [alphafold/model/model.py L122-L141](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L122-L141)

#### Monomer Feature Processing Pipeline

For monomer mode, features are processed through a TensorFlow pipeline ([alphafold/model/features.py L45-L76](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/features.py#L45-L76)

):

```mermaid
flowchart TD

RawFeat["raw_features<br>(NumPy dict)"]
NumRes["Extract num_res<br>from seq_length"]
MakeConfig["make_data_config()[alphafold/model/features.py:28]<br>Set crop_size = num_res"]
DelMatrix["Convert deletion_matrix_int<br>to float32 deletion_matrix"]
TFGraph["Create tf.Graph()[alphafold/model/features.py:60]"]
Convert["np_to_tensor_dict()[alphafold/model/tf/proteins_dataset.py:109]<br>Convert NumPy to TF tensors"]
Reshape["parse_reshape_logic()[alphafold/model/tf/proteins_dataset.py:29]<br>Reshape features to correct dims"]
Process["input_pipeline.process_tensors_from_config()<br>Crop, sample MSA, add random seed"]
Session["tf.Session.run()[alphafold/model/features.py:74]<br>Execute TensorFlow graph"]
Filter["Filter out object dtype<br>features"]
Output["Processed features<br>(NumPy dict)"]

RawFeat --> NumRes
NumRes --> MakeConfig
MakeConfig --> DelMatrix
DelMatrix --> TFGraph
TFGraph --> Convert
Convert --> Reshape
Reshape --> Process
Process --> Session
Session --> Filter
Filter --> Output
```

**Sources:** [alphafold/model/features.py L45-L76](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/features.py#L45-L76)

 [alphafold/model/tf/proteins_dataset.py L109-L131](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/tf/proteins_dataset.py#L109-L131)

### Model Prediction

The `predict` method ([alphafold/model/model.py L153-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L153-L184)

) executes the model:

| Step | Operation | Code Reference |
| --- | --- | --- |
| 1 | Initialize parameters if needed | `self.init_params(feat)` [alphafold/model/model.py L169](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L169-L169) |
| 2 | Log input shapes | `logging.info(...)` [alphafold/model/model.py L170-L173](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L170-L173) |
| 3 | Run JIT-compiled model | `result = self.apply(...)` [alphafold/model/model.py L174](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L174-L174) |
| 4 | Block until outputs ready | `tree.map(lambda x: x.block_until_ready(), result)` [alphafold/model/model.py L179](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L179-L179) |
| 5 | Compute confidence metrics | `get_confidence_metrics(result, multimer_mode)` [alphafold/model/model.py L181](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L181-L181) |
| 6 | Update result dict | `result.update(confidence_metrics)` [alphafold/model/model.py L180-L182](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L180-L182) |
| 7 | Log output shapes | `logging.info(...)` [alphafold/model/model.py L183](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L183-L183) |
| 8 | Return results | `return result` [alphafold/model/model.py L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L184-L184) |

**Sources:** [alphafold/model/model.py L153-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L153-L184)

## Confidence Metric Computation

The `get_confidence_metrics` function ([alphafold/model/model.py L31-L69](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L31-L69)

) post-processes model outputs to compute confidence scores.

```mermaid
flowchart TD

Input["prediction_result<br>(model outputs)"]
pLDDT["confidence.compute_plddt()[alphafold/common/confidence.py]<br>Per-residue confidence"]
CheckPAE["Has predicted_aligned_error?"]
PAE["compute_predicted_aligned_error()[alphafold/common/confidence.py]<br>Pairwise confidence matrix"]
pTM["predicted_tm_score()[alphafold/common/confidence.py]<br>Global confidence metric"]
CheckMultimer["multimer_mode?"]
ipTM["predicted_tm_score(interface=True)[alphafold/common/confidence.py]<br>Interface confidence"]
MonoRank["ranking_confidence =<br>mean(plddt)[alphafold/model/model.py:65]"]
MultiRank["ranking_confidence =<br>0.8 * iptm + 0.2 * ptm[alphafold/model/model.py:59]"]
Output["confidence_metrics dict:<br>plddt, ptm, iptm (multimer),<br>pae, ranking_confidence"]

Input --> pLDDT
Input --> CheckPAE
CheckPAE --> PAE
PAE --> pTM
pTM --> CheckMultimer
CheckMultimer --> ipTM
CheckMultimer --> MonoRank
ipTM --> MultiRank
pLDDT --> Output
PAE --> Output
pTM --> Output
ipTM --> Output
MonoRank --> Output
MultiRank --> Output
CheckPAE --> Output
```

**Sources:** [alphafold/model/model.py L31-L69](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L31-L69)

### Confidence Metrics Summary

| Metric | Computation | Monomer | Multimer | Purpose |
| --- | --- | --- | --- | --- |
| **pLDDT** | `confidence.compute_plddt()` | ✓ | ✓ | Per-residue confidence (0-100) |
| **PAE** | `confidence.compute_predicted_aligned_error()` | ✓* | ✓* | Pairwise alignment error matrix |
| **pTM** | `confidence.predicted_tm_score()` (asym_id=None) | ✓* | ✓* | Global TM-score prediction |
| **ipTM** | `confidence.predicted_tm_score()` (interface=True) | ✗ | ✓* | Interface TM-score (multi-chain) |
| **ranking_confidence** | mean(pLDDT) or 0.8×ipTM + 0.2×pTM | ✓ | ✓ | Used to rank multiple predictions |

*Only available if model has `predicted_aligned_error` head (e.g., `*_ptm` models)

**Sources:** [alphafold/model/model.py L31-L69](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L31-L69)

## Monomer vs Multimer Execution Differences

The execution path diverges based on the `multimer_mode` configuration flag ([alphafold/model/model.py L82](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L82-L82)

).

### Comparison Table

| Aspect | Monomer | Multimer |
| --- | --- | --- |
| **Architecture Module** | `modules.AlphaFold` [alphafold/model/model.py L93](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L93-L93) | `modules_multimer.AlphaFold` [alphafold/model/model.py L87](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L87-L87) |
| **Forward Function Args** | `is_training=False`, `ensemble_representations=True` [alphafold/model/model.py L96-L98](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L96-L98) | `is_training=False` [alphafold/model/model.py L88](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L88-L88) |
| **Feature Processing** | TensorFlow pipeline [alphafold/model/model.py L139](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L139-L139) | No processing [alphafold/model/model.py L137](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L137-L137) |
| **Random Seed Usage** | Controls MSA sampling in processing | Controls MSA sampling in model [alphafold/model/model.py L156](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L156-L156) |
| **ipTM Computation** | Not computed | Computed for interface confidence [alphafold/model/model.py L53-L58](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L53-L58) |
| **Ranking Metric** | `mean(plddt)` [alphafold/model/model.py L65-L67](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L65-L67) | `0.8 * iptm + 0.2 * ptm` [alphafold/model/model.py L59-L61](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L59-L61) |

**Sources:** [alphafold/model/model.py L84-L99](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L84-L99)

 [alphafold/model/model.py L122-L141](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L122-L141)

 [alphafold/model/model.py L51-L67](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L51-L67)

## Evaluation Shape Utility

The `eval_shape` method ([alphafold/model/model.py L143-L151](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L143-L151)

) is a utility for determining output shapes without running the full computation using `jax.eval_shape`.

```mermaid
flowchart TD

Input["eval_shape(feat)"]
Init["init_params(feat)<br>Ensure params exist"]
Log1["Log input shapes"]
Eval["jax.eval_shape(self.apply, ...)[alphafold/model/model.py:149]<br>Trace computation without execution"]
Log2["Log output shapes"]
Return["Return ShapeDtypeStruct"]

Input --> Init
Init --> Log1
Log1 --> Eval
Eval --> Log2
Log2 --> Return
```

**Sources:** [alphafold/model/model.py L143-L151](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L143-L151)

## Usage Pattern

Typical usage of the `RunModel` class follows this pattern:

1. **Instantiate** with configuration and optional pre-trained parameters ([alphafold/model/model.py L75-L81](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L75-L81) ).
2. **Process features** from raw data pipeline output ([alphafold/model/model.py L122-L141](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L122-L141) ).
3. **Predict** to run inference and get results ([alphafold/model/model.py L153-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L153-L184) ).
4. Use confidence metrics for ranking and filtering ([alphafold/model/model.py L31-L69](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L31-L69) ).

The class is designed to be reused for multiple predictions with the same model configuration, amortizing the JIT compilation cost across many inference calls.

**Sources:** [alphafold/model/model.py L72-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L72-L184)