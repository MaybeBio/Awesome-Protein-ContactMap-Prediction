# Model Architecture

> **Relevant source files**
> * [alphafold/model/config.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/config.py)
> * [alphafold/model/features.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/features.py)
> * [alphafold/model/model.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py)
> * [alphafold/model/modules.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py)
> * [alphafold/model/modules_multimer.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py)
> * [alphafold/model/tf/proteins_dataset.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/tf/proteins_dataset.py)

The AlphaFold model architecture is a deep learning system designed to predict protein structures from amino acid sequences with high accuracy. This page provides a high-level deep dive into the neural network architecture, from configuration to execution.

## Core Architecture Overview

AlphaFold transforms sequence and evolutionary information into 3D coordinates through a series of specialized neural network modules. The process is iterative, using a recycling mechanism to refine the structural representation.

```

```

Sources: [alphafold/model/modules.py L134-L213](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L134-L213)

 [alphafold/model/model.py L72-L102](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L72-L102)

## Architecture Components

The architecture is divided into several logical subsystems, each covered in detail in child pages:

### 1. Configuration System

The model's behavior is governed by a hierarchical configuration system. It defines presets for monomer and multimer modes, as well as `CONFIG_DIFFS` that specify variations between model versions (e.g., model_1 vs model_1_ptm).

* **Key Entities**: `MODEL_PRESETS`, `CONFIG_DIFFS`, `BaseConfig`.
* **Details**: For details, see [Configuration System](/google-deepmind/alphafold/4.1-configuration-system).

Sources: [alphafold/model/config.py L30-L118](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/config.py#L30-L118)

 [alphafold/model/base_config.py L20-L30](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/base_config.py#L20-L30)

### 2. Evoformer Stack

The Evoformer is the core processing engine of AlphaFold. It acts on two main representations: the MSA representation and the Pair representation. Through a series of blocks, it performs row-wise and column-wise attention on the MSA, and updates the pair representation using triangle multiplicative updates and triangle attention.

* **Key Entities**: `EmbeddingsAndEvoformer`, `EvoformerIteration`, `TriangleMultiplication`, `OuterProductMean`.
* **Details**: For details, see [Evoformer Stack](/google-deepmind/alphafold/4.2-evoformer-stack).

Sources: [alphafold/model/modules.py L399-L479](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L399-L479)

 [alphafold/model/modules_multimer.py L526-L816](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L526-L816)

### 3. Structure Module

The Structure Module converts the abstract representations from the Evoformer into explicit 3D coordinates. It uses Invariant Point Attention (IPA) to update residue frames and iteratively refines the protein backbone and side-chain geometry.

* **Key Entities**: `StructureModule`, `InvariantPointAttention`, `BackboneUpdate`.
* **Details**: For details, see [Structure Module](/google-deepmind/alphafold/4.3-structure-module).

Sources: [alphafold/model/modules.py L214-L216](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L214-L216)

 [alphafold/model/folding.py L18-L50](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/folding.py#L18-L50)

### 4. Model Execution

Execution is managed by the `RunModel` class, which wraps the Haiku-transformed JAX functions. It handles feature preprocessing, parameter initialization, and the orchestration of the prediction workflow, including JIT compilation for performance.

* **Key Entities**: `RunModel`, `hk.transform`, `jax.jit`, `predict`.
* **Details**: For details, see [Model Execution](/google-deepmind/alphafold/4.4-model-execution).

Sources: [alphafold/model/model.py L72-L184](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L72-L184)

### 5. Model Utilities

The codebase includes various utilities for numerical stability and specialized neural network layers. This includes custom Haiku modules, stable softmax implementations, and context managers for bfloat16 precision.

* **Key Entities**: `stable_softmax`, `bfloat16_context`, `LayerStack`.
* **Details**: For details, see [Model Utilities](/google-deepmind/alphafold/4.5-model-utilities).

Sources: [alphafold/model/utils.py L1-L50](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/utils.py#L1-L50)

 [alphafold/model/layer_stack.py L1-L40](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/layer_stack.py#L1-L40)

## Code Entity Map

The following diagram maps the high-level architecture concepts to the specific classes and functions in the codebase.

```

```

Sources: [alphafold/model/model.py L86-L99](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/model.py#L86-L99)

 [alphafold/model/modules.py L276-L396](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L276-L396)

 [alphafold/model/modules_multimer.py L411-L523](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L411-L523)

## Recycling and Ensembling

AlphaFold utilizes a recycling mechanism (Suppl. Alg. 2) where the output of the structure module and the updated representations are fed back into the start of the model for a fixed number of iterations (controlled by `num_recycle`). In monomer mode, the model can also perform ensembling by averaging representations across multiple stochastic passes.

| Feature | Monomer Behavior | Multimer Behavior |
| --- | --- | --- |
| **Recycling** | 3 iterations by default | Up to 20 iterations with early stopping |
| **Ensembling** | Controlled by `ensemble_representations` | Handled via MSA sampling inside the model |
| **MSA Sampling** | Done in `AlphaFoldIteration` | Managed in `modules_multimer.py` |

Sources: [alphafold/model/modules.py L134-L213](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules.py#L134-L213)

 [alphafold/model/modules_multimer.py L15-L21](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/modules_multimer.py#L15-L21)

 [alphafold/model/config.py L146-L148](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/config.py#L146-L148)