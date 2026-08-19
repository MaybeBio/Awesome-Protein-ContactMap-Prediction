# Configuration Management

> **Relevant source files**
> * [omegafold/__init__.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/__init__.py)
> * [omegafold/config.py](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py)

## Purpose and Scope

This document covers OmegaFold's configuration management system, which defines and manages model parameters across different model variants. The configuration system provides a centralized approach to handling hyperparameters, architectural settings, and component-specific configurations used throughout the neural network pipeline.

For information about how these configurations are applied within the core model components, see [Core Model Components](/HeliXonProtein/OmegaFold/4-core-model-components). For details on the pipeline execution system that consumes these configurations, see [Execution Pipeline](/HeliXonProtein/OmegaFold/6-execution-pipeline).

## Configuration System Overview

OmegaFold uses a hierarchical configuration system based on nested dictionaries that are converted to `argparse.Namespace` objects for convenient attribute access. The system supports multiple model variants with different architectural configurations.

**Configuration Creation Flow**

```mermaid
flowchart TD

A["make_config(model_idx)"]
B["model_idx validation"]
C["Model 1 Configuration"]
D["Model 2 Configuration"]
E["ValueError"]
F["Base Configuration Dict"]
G["_make_config(cfg_dict)"]
H["Recursive Dict Processing"]
I["argparse.Namespace Tree"]
J["Final Configuration Object"]
K["struct_embedder = False"]
L["struct_embedder = True"]

A --> B
B --> C
B --> D
B --> E
C --> F
D --> F
F --> G
G --> H
H --> I
I --> J
K --> C
L --> D
```

Sources: [omegafold/config.py L43-L111](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L43-L111)

The configuration system provides two main functions:

| Function | Purpose | Parameters |
| --- | --- | --- |
| `make_config` | Creates complete model configuration | `model_idx`: 1 or 2 |
| `_make_config` | Recursively converts dict to Namespace | `input_dict`: nested dictionary |

Sources: [omegafold/config.py L32-L41](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L32-L41)

 [omegafold/config.py L43-L111](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L43-L111)

## Model Variants

OmegaFold supports two distinct model variants, differentiated primarily by their structure embedding capabilities:

**Model Variant Comparison**

```mermaid
flowchart TD

A["make_config(model_idx)"]
B["Model Variant Selection"]
C["struct_embedder = False"]
D["Standard Architecture"]
E["struct_embedder = True"]
F["Enhanced Structure Embedding"]
G["Shared Base Configuration"]
H["PLM Configuration"]
I["Geometric Configuration"]
J["Structure Module Configuration"]

A --> B
B --> C
B --> E
G --> C
G --> E
H --> G
I --> G
J --> G

subgraph subGraph1 ["Model 2 (model_idx=2)"]
    E
    F
end

subgraph subGraph0 ["Model 1 (model_idx=1)"]
    C
    D
end
```

Sources: [omegafold/config.py L44-L45](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L44-L45)

 [omegafold/config.py L110](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L110-L110)

## Configuration Parameter Hierarchy

The configuration is organized into logical component groups, each handling specific aspects of the model architecture:

**Configuration Structure**

```mermaid
flowchart TD

A["Configuration Root"]
B["plm"]
C["node_dim / edge_dim"]
D["prev_pos"]
E["binning_configs"]
F["geometric_params"]
G["struct"]
H["struct_embedder"]
B1["alphabet_size: 23"]
B2["node: 1280"]
B3["padding_idx: 21"]
B4["edge: 66"]
B5["attn_dim: 256"]
D1["first_break: 3.25"]
D2["last_break: 20.75"]
D3["num_bins: 16"]
E1["rough_dist_bin"]
E2["dist_bin"]
E3["pos_bin"]
F1["geo_num_blocks: 50"]
F2["attn_c: 32"]
F3["attn_n_head: 8"]
G1["node_dim: 384"]
G2["num_cycle: 8"]
G3["num_head: 12"]

A --> B
A --> C
A --> D
A --> E
A --> F
A --> G
A --> H
B --> B1
B --> B2
B --> B3
B --> B4
B --> B5
D --> D1
D --> D2
D --> D3
E --> E1
E --> E2
E --> E3
F --> F1
F --> F2
F --> F3
G --> G1
G --> G2
G --> G3
```

Sources: [omegafold/config.py L46-L109](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L46-L109)

### Core Configuration Categories

**1. Protein Language Model (PLM) Configuration**

| Parameter | Value | Purpose |
| --- | --- | --- |
| `alphabet_size` | 23 | Vocabulary size for PLM |
| `node` | 1280 | Node dimension |
| `padding_idx` | 21 | Padding token index |
| `edge` | 66 | Edge feature dimension |
| `proj_dim` | 2560 | Projection dimension (1280 * 2) |
| `attn_dim` | 256 | Attention dimension |
| `num_head` | 1 | Number of attention heads |
| `num_relpos` | 129 | Number of relative positions |
| `masked_ratio` | 0.12 | Masking ratio for training |

Sources: [omegafold/config.py L48-L58](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L48-L58)

**2. Geometric Processing Configuration**

| Parameter | Value | Purpose |
| --- | --- | --- |
| `geo_num_blocks` | 50 | Number of geometric blocks |
| `gating` | True | Enable gating mechanism |
| `attn_c` | 32 | Attention channel dimension |
| `attn_n_head` | 8 | Number of attention heads |
| `transition_multiplier` | 4 | Feed-forward expansion factor |
| `activation` | "ReLU" | Activation function |

Sources: [omegafold/config.py L84-L89](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L84-L89)

**3. Structure Module Configuration**

| Parameter | Value | Purpose |
| --- | --- | --- |
| `node_dim` | 384 | Node feature dimension |
| `edge_dim` | 128 | Edge feature dimension |
| `num_cycle` | 8 | Number of recycling cycles |
| `num_transition` | 3 | Number of transition layers |
| `num_head` | 12 | Number of attention heads |
| `num_point_qk` | 4 | Point attention query/key points |
| `num_point_v` | 8 | Point attention value points |
| `num_scalar_qk` | 16 | Scalar attention query/key dimension |
| `num_scalar_v` | 16 | Scalar attention value dimension |

Sources: [omegafold/config.py L94-L108](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L94-L108)

## Distance and Position Binning

The configuration defines multiple binning schemes for different distance and position representations:

**Binning Configuration Structure**

```mermaid
flowchart TD

A["Binning Configurations"]
B["prev_pos"]
C["rough_dist_bin"]
D["dist_bin"]
E["pos_bin"]
B1["first_break: 3.25"]
B2["last_break: 20.75"]
B3["num_bins: 16"]
B4["ignore_index: 0"]
C1["x_min: 3.25"]
C2["x_max: 20.75"]
C3["x_bins: 16"]
D1["x_bins: 64"]
D2["x_min: 2"]
D3["x_max: 65"]
E1["x_bins: 64"]
E2["x_min: -32"]
E3["x_max: 32"]

A --> B
A --> C
A --> D
A --> E
B --> B1
B --> B2
B --> B3
B --> B4
C --> C1
C --> C2
C --> C3
D --> D1
D --> D2
D --> D3
E --> E1
E --> E2
E --> E3
```

Sources: [omegafold/config.py L62-L82](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L62-L82)

## Configuration Integration

The configuration system integrates with the broader OmegaFold architecture through the package's `__init__.py`, making configurations easily accessible to other components:

**Configuration Integration Flow**

```mermaid
flowchart TD

A["omegafold/init.py"]
B["from omegafold.config import make_config"]
C["omegafold.config.make_config"]
D["Model Initialization"]
E["OmegaFold Model"]
F["Component Configuration"]
G["External Usage"]
H["Internal Components"]

A --> B
C --> D
D --> E
D --> F
G --> A
H --> C
```

Sources: [omegafold/__init__.py L23](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/__init__.py#L23-L23)

 [omegafold/__init__.py L24](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/__init__.py#L24-L24)

The configuration object provides dotted-notation access to all parameters, enabling clean integration with PyTorch modules and other system components. For example, `config.plm.node` accesses the PLM node dimension, while `config.struct.num_cycle` retrieves the number of structure refinement cycles.

Sources: [omegafold/config.py L32-L41](https://github.com/HeliXonProtein/OmegaFold/blob/313c873a/omegafold/config.py#L32-L41)