# Training Configuration

> **Relevant source files**
> * [train_monomer_demo.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer_demo.sh)
> * [train_multimer_demo.sh](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer_demo.sh)
> * [unifold/config.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py)

This document covers the training configuration system in Uni-Fold, which defines model architectures, hyperparameters, and training behaviors. The configuration system is implemented in [unifold/config.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py)

 and provides a flexible framework for creating different model variants and training modes.

For information about the actual training scripts and execution, see [Training Scripts](/dptech-corp/Uni-Fold/6.2-training-scripts). For parameter conversion between different model formats, see [Parameter Conversion](/dptech-corp/Uni-Fold/6.3-parameter-conversion).

## Configuration Architecture

The Uni-Fold configuration system is built using `ml_collections.ConfigDict` and provides a hierarchical structure for defining all aspects of model training and inference.

```mermaid
flowchart TD

base_config["base_config()"]
data_config["data"]
model_config["model"]
loss_config["loss"]
globals_config["globals"]
common["common"]
train_mode["train"]
eval_mode["eval"]
predict_mode["predict"]
supervised["supervised"]
input_embedder["input_embedder"]
evoformer_stack["evoformer_stack"]
structure_module["structure_module"]
template["template"]
heads["heads"]
model_config_func["model_config(name, train=False)"]
model_variants["Model Variants"]
model_1["model_1"]
model_2["model_2"]
multimer["multimer"]
model_1_ft["model_1_ft"]
multimer_af2["multimer_af2"]

base_config --> data_config
base_config --> model_config
base_config --> loss_config
base_config --> globals_config
data_config --> common
data_config --> train_mode
data_config --> eval_mode
data_config --> predict_mode
data_config --> supervised
model_config --> input_embedder
model_config --> evoformer_stack
model_config --> structure_module
model_config --> template
model_config --> heads
model_config_func --> base_config
model_config_func --> model_variants
model_variants --> model_1
model_variants --> model_2
model_variants --> multimer
model_variants --> model_1_ft
model_variants --> multimer_af2
```

**Sources:** [unifold/config.py L26-L466](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L26-L466)

 [unifold/config.py L480-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L480-L672)

## Base Configuration Structure

The `base_config()` function defines the foundational configuration that all model variants inherit from. It contains four main sections:

### Data Configuration

The data section controls how input features are processed and prepared for training:

| Section | Purpose | Key Parameters |
| --- | --- | --- |
| `common` | Shared data processing settings | `max_recycling_iters`, `use_templates`, `is_multimer` |
| `train` | Training-specific data processing | `crop=True`, `crop_size=256`, `supervised=True` |
| `eval` | Evaluation data processing | `crop=False`, `num_ensembles=1` |
| `predict` | Inference data processing | `fixed_size=True`, `supervised=False` |

```mermaid
flowchart TD

features["features"]
target_feat["target_feat"]
msa_feat["msa_feat"]
template_feat["template features"]
extra_msa["extra_msa"]
processing["Data Processing"]
masked_msa["masked_msa"]
block_delete_msa["block_delete_msa"]
random_delete_msa["random_delete_msa"]
modes["Processing Modes"]
train["train: crop=True, supervised=True"]
eval["eval: crop=False, supervised=True"]
predict["predict: fixed_size=True, supervised=False"]

features --> target_feat
features --> msa_feat
features --> template_feat
features --> extra_msa
processing --> masked_msa
processing --> block_delete_msa
processing --> random_delete_msa
modes --> train
modes --> eval
modes --> predict
```

**Sources:** [unifold/config.py L29-L227](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L29-L227)

### Model Configuration

The model section defines the neural network architecture:

```mermaid
flowchart TD

model["model"]
input_embedder["input_embedder<br>tf_dim=22, msa_dim=49"]
recycling_embedder["recycling_embedder<br>d_pair=128, d_msa=256"]
template_stack["template<br>template_pair_stack, template_pointwise_attention"]
extra_msa_stack["extra_msa<br>extra_msa_stack: 4 blocks"]
evoformer["evoformer_stack<br>48 blocks, d_msa=256, d_pair=128"]
structure_mod["structure_module<br>8 blocks, d_single=384"]
output_heads["heads<br>plddt, distogram, pae, masked_msa"]

model --> input_embedder
model --> recycling_embedder
model --> template_stack
model --> extra_msa_stack
model --> evoformer
model --> structure_mod
model --> output_heads
```

**Sources:** [unifold/config.py L241-L391](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L241-L391)

### Global Parameters

Global parameters are defined as `mlc.FieldReference` objects that can be shared across the configuration:

| Parameter | Value | Description |
| --- | --- | --- |
| `d_pair` | 128 | Pair representation dimension |
| `d_msa` | 256 | MSA representation dimension |
| `d_single` | 384 | Single representation dimension |
| `max_recycling_iters` | 3 | Maximum recycling iterations |
| `chunk_size` | 4 | Memory management chunk size |

**Sources:** [unifold/config.py L12-L24](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L12-L24)

 [unifold/config.py L228-L240](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L228-L240)

## Model Variants

The `model_config(name, train=False)` function creates specific model variants by modifying the base configuration:

### Monomer Models

```mermaid
flowchart TD

model_1["model_1<br>Base monomer model"]
model_1_ft["model_1_ft<br>Fine-tuning config<br>max_extra_msa=5120<br>crop_size=384"]
model_2["model_2<br>Base model"]
model_2_v2["model_2_v2<br>v2_feature=True<br>gumbel_sample=True"]
model_2_v2_ft["model_2_v2_ft<br>Fine-tuning variant"]
af2_variants["AlphaFold2 Compatible"]
model_1_af2["model_1_af2"]
model_2_af2["model_2_af2"]
model_3_af2["model_3_af2<br>No templates"]
model_4_af2["model_4_af2<br>No templates"]
model_5_af2["model_5_af2<br>No templates"]

model_1 --> model_1_ft
model_2 --> model_2_v2
model_2_v2 --> model_2_v2_ft
af2_variants --> model_1_af2
af2_variants --> model_2_af2
af2_variants --> model_3_af2
af2_variants --> model_4_af2
af2_variants --> model_5_af2
```

**Sources:** [unifold/config.py L515-L585](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L515-L585)

### Multimer Models

Multimer configurations enable protein complex prediction:

```mermaid
flowchart TD

multimer_base["multimer<br>is_multimer=True<br>v2_feature=True"]
multimer_config["Multimer Configuration"]
input_changes["Input Changes<br>tf_dim=21<br>max_extra_msa=1152"]
model_changes["Model Changes<br>separate_kv=True<br>trans_scale_factor=20"]
loss_changes["Loss Changes<br>pae.weight=0.1<br>chain_centre_mass.weight=1.0"]
multimer_ft["multimer_ft<br>Fine-tuning variant"]
multimer_af2["multimer_af2<br>AlphaFold2 compatible"]
multimer_af2_v3["multimer_af2_v3<br>Updated parameters"]

multimer_base --> multimer_config
multimer_config --> input_changes
multimer_config --> model_changes
multimer_config --> loss_changes
multimer_base --> multimer_ft
multimer_base --> multimer_af2
multimer_af2 --> multimer_af2_v3
```

**Sources:** [unifold/config.py L492-L513](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L492-L513)

 [unifold/config.py L586-L665](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L586-L665)

## Training Mode Configurations

Different training phases require different data processing behaviors:

### Training Configuration

```mermaid
flowchart TD

train_config["train"]
data_aug["Data Augmentation<br>crop=True<br>block_delete_msa=True<br>masked_msa_replace_fraction=0.15"]
supervision["Supervision<br>supervised=True<br>use_clamped_fape_prob=1.0"]
memory_opt["Memory Optimization<br>crop_size=256<br>max_msa_clusters=128<br>max_templates=4"]

train_config --> data_aug
train_config --> supervision
train_config --> memory_opt
```

**Sources:** [unifold/config.py L208-L226](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L208-L226)

### Evaluation and Prediction

| Mode | Cropping | Data Augmentation | Supervision | Use Case |
| --- | --- | --- | --- | --- |
| `eval` | No | Minimal | Yes | Validation during training |
| `predict` | No | No | No | Inference on new sequences |

**Sources:** [unifold/config.py L191-L207](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L191-L207)

 [unifold/config.py L176-L190](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L176-L190)

## Configuration Usage in Training

The configuration system integrates with the training scripts through the `--model-name` parameter:

```mermaid
flowchart TD

train_script["train_monomer_demo.sh<br>train_multimer_demo.sh"]
unicore_train["unicore-train"]
model_name["--model-name parameter"]
config_selection["model_config(name, train=True)"]
runtime_config["Runtime Configuration"]
chunk_size_none["chunk_size=None for training"]
inf_eps["inf=3e4, eps=1e-5"]
training_params["Training Parameters"]
optimizer["--optimizer adam"]
lr_scheduler["--lr-scheduler exponential_decay"]
batch_size["--batch-size 1"]
precision["--bf16 --bf16-sr"]

train_script --> unicore_train
unicore_train --> model_name
model_name --> config_selection
config_selection --> runtime_config
runtime_config --> chunk_size_none
runtime_config --> inf_eps
training_params --> optimizer
training_params --> lr_scheduler
training_params --> batch_size
training_params --> precision
```

**Sources:** [train_monomer_demo.sh L7-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer_demo.sh#L7-L15)

 [train_multimer_demo.sh L7-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer_demo.sh#L7-L15)

 [unifold/config.py L668-L672](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L668-L672)

### Key Training Parameters

The training scripts demonstrate how configuration parameters map to command-line arguments:

| Script Parameter | Purpose | Configuration Impact |
| --- | --- | --- |
| `--task af2` | Task type | Determines data loading behavior |
| `--loss af2/afm` | Loss function | Monomer vs multimer loss |
| `--model-name` | Model variant | Selects specific configuration |
| `--bf16` | Mixed precision | Memory and speed optimization |

**Sources:** [train_monomer_demo.sh L9-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_monomer_demo.sh#L9-L15)

 [train_multimer_demo.sh L9-L15](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/train_multimer_demo.sh#L9-L15)

## Configuration Utilities

The configuration system includes utility functions for dynamic modification:

### Recursive Configuration Setting

The `recursive_set()` function enables bulk parameter updates across nested configuration dictionaries:

```python
def recursive_set(c: mlc.ConfigDict, key: str, value: Any, ignore: str = None)
```

This function is used extensively in model variants to apply consistent changes across the entire configuration hierarchy.

**Sources:** [unifold/config.py L469-L477](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/config.py#L469-L477)