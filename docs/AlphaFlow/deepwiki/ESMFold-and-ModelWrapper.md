# ESMFold and ModelWrapper

> **Relevant source files**
> * [alphaflow/model/esmfold.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py)
> * [alphaflow/model/wrapper.py](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py)

This document covers the ESMFold model implementation and the ModelWrapper training framework that provides the core training and inference capabilities for both ESMFold and AlphaFold models in the AlphaFlow system.

For information about the AlphaFold model architecture, see [AlphaFold Architecture](#5.3). For details about dataset classes used with these models, see [Dataset Classes](/bjing2016/alphaflow/5.2-dataset-classes).

## ESMFold Architecture

The `ESMFold` class extends Meta's ESMFold with diffusion capabilities for ensemble generation. It combines the ESM protein language model with a folding trunk and specialized input processing for handling noisy distance inputs during diffusion training.

### Core Components

The ESMFold architecture consists of several key components:

```mermaid
flowchart TD

ESM["esm (Frozen)<br>Pretrained ESM2 Model"]
ESM_DICT["esm_dict<br>Amino Acid Alphabet"]
ESM_S_COMBINE["esm_s_combine<br>Layer Combination Weights"]
ESM_S_MLP["esm_s_mlp<br>Sequence Feature MLP"]
EMBEDDING["embedding<br>Residue Type Embedding"]
INPUT_PAIR_EMB["input_pair_embedding<br>Distance Pair Embedding"]
INPUT_TIME_PROJ["input_time_projection<br>GaussianFourierProjection"]
INPUT_TIME_EMB["input_time_embedding<br>Time Embedding"]
INPUT_PAIR_STACK["input_pair_stack<br>InputPairStack"]
EXTRA_PAIR_EMB["extra_input_pair_embedding<br>Template Pair Embedding"]
EXTRA_PAIR_STACK["extra_input_pair_stack<br>Extra InputPairStack"]
TRUNK["trunk<br>FoldingTrunk"]
DIST_HEAD["distogram_head<br>Distance Prediction"]
LDDT_HEAD["lddt_head<br>Confidence Prediction"]

ESM --> ESM_S_COMBINE
ESM_S_MLP --> TRUNK
EMBEDDING --> TRUNK
INPUT_PAIR_STACK --> TRUNK
EXTRA_PAIR_STACK --> INPUT_PAIR_STACK

subgraph subGraph4 ["Core Architecture"]
    TRUNK
    DIST_HEAD
    LDDT_HEAD
    TRUNK --> DIST_HEAD
    TRUNK --> LDDT_HEAD
end

subgraph subGraph3 ["Extra Input (Optional)"]
    EXTRA_PAIR_EMB
    EXTRA_PAIR_STACK
    EXTRA_PAIR_EMB --> EXTRA_PAIR_STACK
end

subgraph subGraph2 ["Pair Processing"]
    INPUT_PAIR_EMB
    INPUT_TIME_PROJ
    INPUT_TIME_EMB
    INPUT_PAIR_STACK
    INPUT_PAIR_EMB --> INPUT_PAIR_STACK
    INPUT_TIME_PROJ --> INPUT_TIME_EMB
    INPUT_TIME_EMB --> INPUT_PAIR_STACK
end

subgraph subGraph1 ["Sequence Processing"]
    ESM_S_COMBINE
    ESM_S_MLP
    EMBEDDING
    ESM_S_COMBINE --> ESM_S_MLP
end

subgraph subGraph0 ["ESM Language Model"]
    ESM
    ESM_DICT
end
```

Sources: [alphaflow/model/esmfold.py L49-L124](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L49-L124)

### Key Methods

| Method | Purpose | Key Features |
| --- | --- | --- |
| `__init__` | Initialize model components | Sets up ESM model, embeddings, trunk, and heads |
| `forward` | Main forward pass | Handles noisy inputs, time conditioning, optional extra inputs |
| `infer` | Sequence-to-structure inference | Batch processing, multimer support, residue indexing |
| `_compute_language_model_representations` | Extract ESM features | BOS/EOS token handling, attention extraction |
| `_get_input_pair_embeddings` | Process distance inputs | Distance binning, pair embedding, stack processing |

Sources: [alphaflow/model/esmfold.py L125-L437](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L125-L437)

## ModelWrapper Training Framework

The `ModelWrapper` class provides a PyTorch Lightning-based training framework that supports diffusion training, distillation, and exponential moving averages. It serves as the base class for both `ESMFoldWrapper` and `AlphaFoldWrapper`.

### Training Architecture

```mermaid
flowchart TD

TRAINING_STEP["training_step()<br>Main Training Logic"]
VALIDATION_STEP["validation_step()<br>Validation Logic"]
INFERENCE["inference()<br>Diffusion Inference"]
ADD_NOISE["_add_noise()<br>Noise Addition"]
ESMFOLD_MODEL["model: ESMFold"]
ESMFOLD_TEACHER["teacher: ESMFold<br>(Distillation)"]
ESMFOLD_LOSS["loss: AlphaFoldLoss<br>(esmfold=True)"]
ALPHAFOLD_MODEL["model: AlphaFold"]
ALPHAFOLD_TEACHER["teacher: AlphaFold<br>(Distillation)"]
ALPHAFOLD_LOSS["loss: AlphaFoldLoss"]
EMA["ema: ExponentialMovingAverage"]
HARMONIC_PRIOR["harmonic_prior: HarmonicPrior"]
OPTIMIZER["configure_optimizers()<br>Adam + AlphaFoldLRScheduler"]

TRAINING_STEP --> ESMFOLD_MODEL
TRAINING_STEP --> ALPHAFOLD_MODEL
INFERENCE --> ESMFOLD_MODEL
INFERENCE --> ALPHAFOLD_MODEL
ADD_NOISE --> HARMONIC_PRIOR
TRAINING_STEP --> EMA
ESMFOLD_LOSS --> TRAINING_STEP
ALPHAFOLD_LOSS --> TRAINING_STEP

subgraph subGraph3 ["Training Components"]
    EMA
    HARMONIC_PRIOR
    OPTIMIZER
end

subgraph AlphaFoldWrapper ["AlphaFoldWrapper"]
    ALPHAFOLD_MODEL
    ALPHAFOLD_TEACHER
    ALPHAFOLD_LOSS
end

subgraph ESMFoldWrapper ["ESMFoldWrapper"]
    ESMFOLD_MODEL
    ESMFOLD_TEACHER
    ESMFOLD_LOSS
end

subgraph subGraph0 ["Base ModelWrapper"]
    TRAINING_STEP
    VALIDATION_STEP
    INFERENCE
    ADD_NOISE
    TRAINING_STEP --> ADD_NOISE
    VALIDATION_STEP --> INFERENCE
end
```

Sources: [alphaflow/model/wrapper.py L51-L515](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L51-L515)

### Training Process Flow

The training process implements diffusion-based ensemble generation with optional self-conditioning and distillation:

```mermaid
flowchart TD

DIST_START["disillation_training_step()"]
TEACHER_FORWARD["Teacher forward pass<br>(multiple steps)"]
STUDENT_FORWARD["Student forward pass<br>(single step)"]
DIST_LOSS["Distillation loss"]
START["training_step()"]
NOISE_CHECK["noise_prob > random?"]
ADD_NOISE_STEP["_add_noise()<br>Add diffusion noise"]
SELF_COND_CHECK["self_cond_prob > random?"]
SELF_COND_FORWARD["Forward pass<br>(no_grad)"]
MAIN_FORWARD["Forward pass<br>(with prev_outputs)"]
LOSS_CALC["Loss calculation"]
METRICS["Validation metrics"]

subgraph subGraph1 ["Distillation Training"]
    DIST_START
    TEACHER_FORWARD
    STUDENT_FORWARD
    DIST_LOSS
    DIST_START --> TEACHER_FORWARD
    TEACHER_FORWARD --> STUDENT_FORWARD
    STUDENT_FORWARD --> DIST_LOSS
end

subgraph subGraph0 ["Standard Training"]
    START
    NOISE_CHECK
    ADD_NOISE_STEP
    SELF_COND_CHECK
    SELF_COND_FORWARD
    MAIN_FORWARD
    LOSS_CALC
    METRICS
    START --> NOISE_CHECK
    NOISE_CHECK --> ADD_NOISE_STEP
    NOISE_CHECK --> SELF_COND_CHECK
    ADD_NOISE_STEP --> SELF_COND_CHECK
    SELF_COND_CHECK --> SELF_COND_FORWARD
    SELF_COND_CHECK --> MAIN_FORWARD
    SELF_COND_FORWARD --> MAIN_FORWARD
    MAIN_FORWARD --> LOSS_CALC
    LOSS_CALC --> METRICS
end
```

Sources: [alphaflow/model/wrapper.py L52-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L52-L175)

## Diffusion Noise Addition

The `_add_noise` method implements the core diffusion process by interpolating between ground truth and random structures:

### Noise Addition Process

| Component | Purpose | Implementation |
| --- | --- | --- |
| `HarmonicPrior` | Generate random structures | Samples from harmonic distribution |
| `rmsdalign` | Align structures | RMSD-based structural alignment |
| Time sampling | Control noise level | `t ~ Uniform(0,1)` |
| Interpolation | Mix clean/noisy | `(1-t) * clean + t * noisy` |

The process creates noisy pseudo-beta distances for training:

```python
# Simplified noise addition logic from _add_noise()noisy = harmonic_prior.sample()noisy = rmsdalign(ground_truth, noisy)t = random_uniform()noisy_beta = (1 - t) * ground_truth + t * noisy
```

Sources: [alphaflow/model/wrapper.py L52-L71](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L52-L71)

## Inference Pipeline

The `inference` method implements iterative denoising for structure generation:

### Diffusion Schedule

```mermaid
flowchart TD

T1["t=1.0<br>Pure Noise"]
T2["t=0.75<br>Partial Denoise"]
T3["t=0.5<br>Moderate Denoise"]
T4["t=0.25<br>High Denoise"]
T5["t=0.1<br>Near Final"]
T6["t=0.0<br>Final Structure"]

T1 --> T2
T2 --> T3
T3 --> T4
T4 --> T5
T5 --> T6
```

### Self-Conditioning Flow

```mermaid
flowchart TD

INIT_NOISE["Initialize<br>noisy ~ HarmonicPrior()"]
STEP["Forward pass<br>model(batch, prev_outputs)"]
EXTRACT["Extract pseudo_beta<br>from predictions"]
ALIGN["RMSD align<br>prediction to noise"]
INTERPOLATE["Interpolate<br>(s/t)*noisy + (1-s/t)*pred"]
UPDATE_BATCH["Update batch<br>with new distances"]
NEXT_STEP["More steps?"]
RETURN["Return structures"]
SELF_COND["prev_outputs<br>(Self-conditioning)"]

NEXT_STEP --> RETURN
STEP --> SELF_COND
SELF_COND --> STEP

subgraph subGraph0 ["Iterative Denoising"]
    INIT_NOISE
    STEP
    EXTRACT
    ALIGN
    INTERPOLATE
    UPDATE_BATCH
    NEXT_STEP
    INIT_NOISE --> STEP
    STEP --> EXTRACT
    EXTRACT --> ALIGN
    ALIGN --> INTERPOLATE
    INTERPOLATE --> UPDATE_BATCH
    UPDATE_BATCH --> NEXT_STEP
    NEXT_STEP --> STEP
end
```

Sources: [alphaflow/model/wrapper.py L350-L392](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L350-L392)

## Key Configuration Parameters

### ESMFold Configuration

| Parameter | Purpose | Location |
| --- | --- | --- |
| `esm_type` | ESM model variant | [alphaflow/model/esmfold.py L57](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L57-L57) |
| `trunk` | Folding trunk config | [alphaflow/model/esmfold.py L112](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L112-L112) |
| `input_pair_embedder` | Distance embedding config | [alphaflow/model/esmfold.py L77-L92](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L77-L92) |
| `use_esm_attn_map` | Use ESM attention | [alphaflow/model/esmfold.py L153](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/esmfold.py#L153-L153) |

### Training Configuration

| Parameter | Purpose | Default/Usage |
| --- | --- | --- |
| `noise_prob` | Probability of adding noise | Training control |
| `self_cond_prob` | Self-conditioning probability | Training enhancement |
| `extra_input_prob` | Template input probability | Template training |
| `distillation` | Enable distillation training | Knowledge transfer |
| `no_ema` | Disable exponential moving average | Weight averaging |

Sources: [alphaflow/model/wrapper.py L126-L175](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L126-L175)

## Validation and Metrics

The validation process generates ensemble predictions and computes structural metrics:

### Validation Metrics

| Metric Type | Metrics | Purpose |
| --- | --- | --- |
| Reference Comparison | RMSD, GDT-TS, GDT-HA, LDDT | Compare to ground truth |
| Self Comparison | Self-RMSD, Self-LDDT | Measure ensemble diversity |
| Single Structure | First prediction metrics | Assess initial quality |

The validation generates multiple samples per input and computes both reference and self-consistency metrics to evaluate both accuracy and ensemble quality.

Sources: [alphaflow/model/wrapper.py L177-L229](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L177-L229)

 [alphaflow/model/wrapper.py L394-L446](https://github.com/bjing2016/alphaflow/blob/02dc0376/alphaflow/model/wrapper.py#L394-L446)