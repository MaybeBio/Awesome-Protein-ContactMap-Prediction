# User Interfaces

> **Relevant source files**
> * [README.md](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1)
> * [notebooks/unifold.ipynb](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb)
> * [unifold/inference.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py)

This document covers the various user interfaces available for interacting with Uni-Fold, including command-line tools, interactive notebooks, and specialized interfaces for different prediction types. For information about the underlying model architecture, see [Model Architecture](/dptech-corp/Uni-Fold/5-model-architecture). For details on data processing pipelines, see [Data Pipeline](/dptech-corp/Uni-Fold/4-data-pipeline).

## Interface Overview

Uni-Fold provides multiple user interfaces to accommodate different use cases and user preferences:

* **Command Line Interface**: Primary interface for batch processing and production use
* **Colab Notebook Interface**: Interactive web-based interface for experimental use
* **UF-Symmetry Interface**: Specialized interface for symmetric protein complex prediction
* **Docker Container Interface**: Containerized deployment for reproducible environments

```mermaid
flowchart TD

A["run_unifold.sh"]
B["unifold.ipynb"]
C["run_uf_symmetry.sh"]
D["Docker Container"]
E["inference.py"]
F["homo_search.py"]
G["UnifoldDataset"]
H["AlphaFold Model"]
I["model_config()"]
J["Parameter Files"]
K["Feature Processing"]

A --> E
A --> F
B --> E
B --> F
C --> E
C --> F
I --> H
J --> H
K --> G

subgraph subGraph2 ["Configuration & Parameters"]
    I
    J
    K
end

subgraph subGraph1 ["Core Interface Components"]
    E
    F
    G
    H
    E --> G
    F --> G
    G --> H
end

subgraph subGraph0 ["Entry Points"]
    A
    B
    C
    D
    D --> A
end
```

Sources: [README.md L1-L344](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L1-L344)

 [notebooks/unifold.ipynb L1-L350](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L1-L350)

 [unifold/inference.py L1-L267](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L1-L267)

## Interface Entry Points and Data Flow

The following diagram shows how different interfaces process input sequences through to final structure predictions:

```mermaid
flowchart TD

A["FASTA Sequence"]
B["Symmetry Group"]
C["Configuration Parameters"]
D["Interface Selection"]
E["run_unifold.sh"]
F["unifold.ipynb Colab"]
G["run_uf_symmetry.sh"]
H["homo_search.py"]
I["get_msa_and_templates()"]
J["MMseqs2 Service"]
K["Feature Processing"]
L["load_feature_for_one_target()"]
M["AlphaFold.forward()"]
N["automatic_chunk_size()"]
O["protein.from_prediction()"]
P["PDB Files"]
Q["Confidence Scores"]
R["Visualization"]

A --> D
B --> D
C --> D
E --> H
F --> I
F --> J
G --> H
K --> L
M --> O
F --> R

subgraph subGraph4 ["Output Generation"]
    O
    P
    Q
    R
    O --> P
    O --> Q
end

subgraph subGraph3 ["Model Execution"]
    L
    M
    N
    L --> M
    M --> N
end

subgraph subGraph2 ["Feature Generation"]
    H
    I
    J
    K
    H --> K
    I --> K
    J --> K
end

subgraph subGraph1 ["Interface Dispatch"]
    D
    E
    F
    G
    D --> E
    D --> F
    D --> G
end

subgraph subGraph0 ["Input Layer"]
    A
    B
    C
end
```

Sources: [README.md L125-L141](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L125-L141)

 [notebooks/unifold.ipynb L132-L240](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L132-L240)

 [unifold/inference.py L49-L74](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L49-L74)

 [unifold/inference.py L76-L199](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L76-L199)

## Core Interface Components

### Command Line Interface Architecture

The primary CLI interface `run_unifold.sh` orchestrates the complete prediction pipeline:

| Component | File Path | Function | Purpose |
| --- | --- | --- | --- |
| Main Entry Point | `run_unifold.sh` | Shell script coordination | Orchestrates MSA search and inference |
| Homology Search | `homo_search.py` | Database searching | Generates MSAs and templates |
| Model Inference | `inference.py` | `main()` | Loads models and runs predictions |
| Feature Loading | `inference.py` | `load_feature_for_one_target()` | Processes features for model input |
| Memory Management | `inference.py` | `automatic_chunk_size()` | Optimizes memory usage based on sequence length |

```mermaid
flowchart TD

A["run_unifold.sh"]
B["homo_search.py"]
C["inference.py main()"]
D["load_feature_for_one_target()"]
E["AlphaFold.forward()"]
F["protein.from_prediction()"]
G["model_name"]
H["param_path"]
I["data_dir"]
J["target_name"]
K["max_recycling_iters"]

G --> C
H --> C
I --> D
J --> D
K --> C

subgraph subGraph1 ["Key Parameters"]
    G
    H
    I
    J
    K
end

subgraph subGraph0 ["CLI Workflow"]
    A
    B
    C
    D
    E
    F
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
end
```

Sources: [README.md L125-L141](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L125-L141)

 [unifold/inference.py L77-L199](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L77-L199)

 [unifold/inference.py L202-L266](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L202-L266)

### Colab Notebook Interface Components

The interactive notebook interface provides a streamlined web-based experience:

```mermaid
flowchart TD

A["Parameter Configuration"]
B["Environment Setup"]
C["MSA Generation"]
D["Model Inference"]
E["Structure Visualization"]
F["Result Download"]
G["validate_input()"]
H["get_msa_and_templates()"]
I["colab_inference()"]
J["colab_plot()"]
K["MMseqs2 Server"]
L["ColabFold Service"]

A --> G
B --> H
C --> H
C --> K
C --> L
D --> I
E --> J
J --> F

subgraph subGraph2 ["External Services"]
    K
    L
end

subgraph subGraph1 ["Core Functions"]
    G
    H
    I
    J
    H --> I
    I --> J
end

subgraph subGraph0 ["Notebook Cells"]
    A
    B
    C
    D
    E
    F
end
```

Sources: [notebooks/unifold.ipynb L36-L72](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L36-L72)

 [notebooks/unifold.ipynb L132-L240](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L132-L240)

 [notebooks/unifold.ipynb L251-L275](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L251-L275)

 [notebooks/unifold.ipynb L286-L296](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L286-L296)

### UF-Symmetry Interface Specialization

The UF-Symmetry interface extends the base CLI with symmetric complex prediction capabilities:

| Parameter | Description | Example Values |
| --- | --- | --- |
| `symmetry_group` | Symmetry type specification | `C3`, `C4`, `D2`, etc. |
| Input Requirements | Asymmetric unit only | Single chain representing unit |
| Model Selection | Specialized symmetry parameters | `uf_symmetry_params_2022-09-06.tar.gz` |
| Assembly Generation | Symmetry expansion | Automatic complex generation |

```mermaid
flowchart TD

A["run_uf_symmetry.sh"]
B["Asymmetric Unit FASTA"]
C["Symmetry Group Specification"]
D["Feature Processing"]
E["UFSymmetry Model"]
F["Assembly Generation"]
G["Full Complex PDB"]
H["Cyclic Groups (C2, C3, C4...)"]
I["Dihedral Groups (D2, D3, D4...)"]
J["Icosahedral (I)"]
K["Octahedral (O)"]
L["Tetrahedral (T)"]

C --> H
C --> I
C --> J
C --> K
C --> L

subgraph subGraph1 ["Symmetry Types"]
    H
    I
    J
    K
    L
end

subgraph subGraph0 ["UF-Symmetry Workflow"]
    A
    B
    C
    D
    E
    F
    G
    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
end
```

Sources: [README.md L260-L282](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/README.md?plain=1#L260-L282)

 [notebooks/unifold.ipynb L49-L54](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/notebooks/unifold.ipynb#L49-L54)

## Interface Configuration and Parameters

### Model Configuration System

All interfaces utilize the centralized configuration system defined in `unifold/config.py`:

```mermaid
flowchart TD

A["model_config()"]
B["Model Name Selection"]
C["Parameter Loading"]
D["Runtime Configuration"]
E["model_1"]
F["model_2"]
G["model_2_ft"]
H["multimer"]
I["multimer_ft"]
J["max_recycling_iters"]
K["num_ensembles"]
L["chunk_size"]
M["bf16"]

B --> E
B --> F
B --> G
B --> H
B --> I
D --> J
D --> K
D --> L
D --> M

subgraph subGraph2 ["Runtime Parameters"]
    J
    K
    L
    M
end

subgraph subGraph1 ["Model Variants"]
    E
    F
    G
    H
    I
end

subgraph subGraph0 ["Configuration Flow"]
    A
    B
    C
    D
    A --> B
    B --> C
    C --> D
end
```

Sources: [unifold/inference.py L78-L96](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L78-L96)

 [unifold/inference.py L127-L133](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L127-L133)

### Memory Management and Performance Optimization

The `automatic_chunk_size()` function optimizes memory usage based on hardware capabilities:

| Sequence Length Range | Chunk Size | Block Size | Memory Factor |
| --- | --- | --- | --- |
| < 1024 * factor | 256 | None | Based on GPU memory |
| < 2048 * factor | 128 | None | Adjusted for bf16 |
| < 3072 * factor | 64 | None | Dynamic scaling |
| < 4096 * factor | 32 | 512 | Conservative approach |
| ≥ 4096 * factor | 4 | 256 | Maximum memory efficiency |

Sources: [unifold/inference.py L20-L47](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L20-L47)

 [unifold/inference.py L125-L133](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/inference.py#L125-L133)