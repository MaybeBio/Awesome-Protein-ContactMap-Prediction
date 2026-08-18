# AlphaFold2 Notebooks

> **Relevant source files**
> * [AlphaFold2.ipynb](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb)
> * [Boltz1.ipynb](https://github.com/sokrypton/ColabFold/blob/0c788a0e/Boltz1.ipynb)
> * [batch/AlphaFold2_batch.ipynb](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb)

## Purpose and Scope

The AlphaFold2 Notebooks provide interactive Google Colab interfaces for protein structure prediction using the ColabFold system. These notebooks serve as the primary user-facing entry points, offering form-based interfaces for configuring and executing protein folding predictions. This document covers the two main AlphaFold2 notebooks: the standard prediction notebook (`AlphaFold2.ipynb`) and the batch processing notebook (`batch/AlphaFold2_batch.ipynb`).

For information about advanced modeling features and experimental options, see [Advanced AlphaFold2 Notebooks](/sokrypton/ColabFold/3.2.2-advanced-alphafold2-notebooks). For alternative protein folding models, see [Alternative Model Notebooks](/sokrypton/ColabFold/3.2.3-alternative-model-notebooks).

## Notebook Architecture Overview

```mermaid
flowchart TD

A["AlphaFold2.ipynb"]
B["AlphaFold2_batch.ipynb"]
A1["Input Forms & Parameters"]
A2["Dependency Installation"]
A3["Configuration Validation"]
A4["Execution Controls"]
C1["colabfold.batch.run"]
C2["colabfold.batch.get_queries"]
C3["colabfold.download.download_alphafold_params"]
C4["colabfold.utils.setup_logging"]
D1["3D Visualization (py3Dmol)"]
D2["Result Plotting"]
D3["File Download/Drive Upload"]

A --> A1
A --> A2
A --> A3
A --> A4
B --> A1
B --> A2
B --> A4
A1 --> C2
A2 --> C3
A3 --> C4
A4 --> C1
C1 --> D1
C1 --> D2
C1 --> D3

subgraph subGraph3 ["Output Generation"]
    D1
    D2
    D3
end

subgraph subGraph2 ["Core Integration"]
    C1
    C2
    C3
    C4
end

subgraph subGraph1 ["User Interface Layer"]
    A1
    A2
    A3
    A4
end

subgraph subGraph0 ["Google Colab Environment"]
    A
    B
end
```

**Sources:** [AlphaFold2.ipynb L1-L643](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L1-L643)

 [batch/AlphaFold2_batch.ipynb L1-L275](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L1-L275)

## Single Sequence Prediction Notebook

The main `AlphaFold2.ipynb` notebook provides an interactive interface for predicting individual protein structures or complexes.

### Input Configuration

```mermaid
flowchart TD

A["query_sequence"]
B["jobname"]
C["num_relax"]
D["template_mode"]
E["add_hash()"]
F["check()"]
G["queries_path generation"]
H["jobname directory"]
I["jobname.csv"]
J["template directory"]

A --> E
B --> E
G --> H
D --> J

subgraph subGraph2 ["File System"]
    H
    I
    J
    H --> I
end

subgraph subGraph1 ["Processing Logic"]
    E
    F
    G
    E --> F
    F --> G
end

subgraph subGraph0 ["Input Parameters"]
    A
    B
    C
    D
end
```

The notebook accepts protein sequences with `:` as chain separators for complex modeling [AlphaFold2.ipynb L78](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L78-L78)

 The `add_hash` function [AlphaFold2.ipynb L74-L75](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L74-L75)

 generates unique job identifiers by combining the base job name with a hash of the query sequence.

**Sources:** [AlphaFold2.ipynb L74-L130](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L74-L130)

### MSA and Template Configuration

```mermaid
flowchart TD

A["msa_mode"]
A1["MMseqs2 (UniRef+Environmental)"]
A2["MMseqs2 (UniRef only)"]
A3["single_sequence"]
A4["custom"]
B["template_mode"]
B1["none"]
B2["pdb100"]
B3["custom"]
C["a3m_file"]
D["custom_template_path"]
E["files.upload()"]

A4 --> E
B3 --> E

subgraph subGraph2 ["File Generation"]
    C
    D
    E
    E --> C
    E --> D
end

subgraph subGraph1 ["Template Processing"]
    B
    B1
    B2
    B3
    B --> B1
    B --> B2
    B --> B3
end

subgraph subGraph0 ["MSA Mode Selection"]
    A
    A1
    A2
    A3
    A4
    A --> A1
    A --> A2
    A --> A3
    A --> A4
end
```

The MSA configuration logic [AlphaFold2.ipynb L195-L222](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L195-L222)

 determines the alignment file path based on the selected mode. Custom template uploads are processed through Google Colab's `files.upload()` interface [AlphaFold2.ipynb L120](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L120-L120)

**Sources:** [AlphaFold2.ipynb L83-L84](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L83-L84)

 [AlphaFold2.ipynb L114-L127](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L114-L127)

 [AlphaFold2.ipynb L190-L227](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L190-L227)

### Prediction Execution Pipeline

```mermaid
flowchart TD

A["download_alphafold_params()"]
B["get_queries()"]
C["set_model_type()"]
D["setup_logging()"]
E["input_features_callback"]
F["prediction_callback"]
G["plot_msa_v2()"]
H["plot_protein()"]
I["colabfold.batch.run()"]
J["results dict"]
K["results_zip"]
L["os.system zip command"]

A --> I
B --> I
C --> I
D --> I
I --> E
I --> F
I --> J

subgraph subGraph3 ["Output Processing"]
    J
    K
    L
    J --> K
    K --> L
end

subgraph subGraph2 ["Main Execution"]
    I
end

subgraph subGraph1 ["Callback Functions"]
    E
    F
    G
    H
    E --> G
    F --> H
end

subgraph subGraph0 ["Setup Phase"]
    A
    B
    C
    D
end
```

The main prediction execution [AlphaFold2.ipynb L355-L387](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L355-L387)

 calls `colabfold.batch.run` with comprehensive configuration parameters and callback functions for real-time visualization.

**Sources:** [AlphaFold2.ipynb L300-L396](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L300-L396)

## Batch Processing Notebook

The `AlphaFold2_batch.ipynb` notebook handles multiple protein predictions from Google Drive directories [batch/AlphaFold2_batch.ipynb L48-L53](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L48-L53)

### Batch Configuration Structure

| Parameter | Type | Purpose |
| --- | --- | --- |
| `input_dir` | string | Google Drive path containing FASTA files or MSAs |
| `result_dir` | string | Output directory for results |
| `num_models` | int | Number of models to generate (1-5) |
| `num_recycles` | int | Recycling iterations (1-48) |
| `stop_at_score` | float | Early stopping threshold |
| `use_templates` | boolean | Enable template search |
| `zip_results` | boolean | Archive output files |

**Sources:** [batch/AlphaFold2_batch.ipynb L91-L108](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L91-L108)

### Batch Processing Flow

```mermaid
flowchart TD

D["pip install colabfold"]
E["conda install dependencies"]
F["python -m colabfold.download"]
A["drive.mount()"]
B["input_dir"]
C["result_dir"]
G["get_queries(input_dir)"]
H["batch.run()"]
I["setup_logging()"]

B --> G
C --> I

subgraph subGraph2 ["Batch Execution"]
    G
    H
    I
    G --> H
    I --> H
end

subgraph subGraph0 ["Google Drive Integration"]
    A
    B
    C
    A --> B
    A --> C
end

subgraph subGraph1 ["Dependency Management"]
    D
    E
    F
    D --> E
    E --> F
end
```

The batch notebook [batch/AlphaFold2_batch.ipynb L175-L210](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L175-L210)

 processes all input files through a single `colabfold.batch.run` call with standardized parameters.

**Sources:** [batch/AlphaFold2_batch.ipynb L71-L213](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L71-L213)

## Advanced Configuration Parameters

### Model and Sampling Settings

```mermaid
flowchart TD

A["model_type"]
A1["auto"]
A2["alphafold2_ptm"]
A3["alphafold2_multimer_v3"]
A4["deepfold_v1"]
B["num_seeds"]
C["use_dropout"]
D["max_msa"]
E["num_recycles"]
F["set_model_type()"]
G["use_cluster_profile"]

A --> F
B --> G
C --> G
D --> G
E --> F

subgraph subGraph2 ["Processing Logic"]
    F
    G
end

subgraph subGraph1 ["Sampling Parameters"]
    B
    C
    D
    E
end

subgraph subGraph0 ["Model Configuration"]
    A
    A1
    A2
    A3
    A4
end
```

The advanced settings [AlphaFold2.ipynb L235-L258](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L235-L258)

 control model selection, sampling behavior, and uncertainty estimation through dropout and MSA subsampling.

**Sources:** [AlphaFold2.ipynb L234-L287](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L234-L287)

### Pairing and Complex Prediction

| Parameter | Values | Description |
| --- | --- | --- |
| `pair_mode` | `unpaired_paired`, `paired`, `unpaired` | MSA pairing strategy |
| `pairing_strategy` | `greedy`, `complete` | Taxonomic matching approach |
| `calc_extra_ptm` | boolean | Additional pTM calculations |

**Sources:** [AlphaFold2.ipynb L192-L247](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L192-L247)

 [batch/AlphaFold2_batch.ipynb L63](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L63-L63)

## Integration with Core System

```mermaid
flowchart TD

A["User Interface Forms"]
B["Parameter Validation"]
C["File Management"]
D["colabfold.batch.get_queries"]
E["colabfold.batch.run"]
F["colabfold.batch.set_model_type"]
G["colabfold.download"]
H["colabfold.utils.setup_logging"]
I["colabfold.plot.plot_msa_v2"]
J["colabfold.colabfold.plot_protein"]
K["py3Dmol"]
L["matplotlib"]
M["google.colab.files"]

C --> D
E --> G
E --> H
E --> I
E --> J
I --> L
J --> L
C --> M
J --> K

subgraph subGraph3 ["External Dependencies"]
    K
    L
    M
end

subgraph subGraph2 ["Utility Integration"]
    G
    H
    I
    J
end

subgraph subGraph1 ["Batch Module Integration"]
    D
    E
    F
    D --> E
    E --> F
end

subgraph subGraph0 ["Notebook Layer"]
    A
    B
    C
    A --> B
    B --> C
end
```

The notebooks serve as configuration interfaces that translate user inputs into calls to the core `colabfold.batch.run` system [AlphaFold2.ipynb L355-L387](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L355-L387)

 with integrated visualization and file management capabilities.

**Sources:** [AlphaFold2.ipynb L300-L302](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L300-L302)

 [batch/AlphaFold2_batch.ipynb L179-L181](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L179-L181)

## Dependency Installation and Environment Setup

### Installation Cell Structure

```mermaid
flowchart TD

H["HH_READY (templates)"]
J["hhsuite installation"]
I["AMBER_READY (relaxation)"]
K["openmm installation"]
E["CONDA_READY check"]
F["Miniforge download"]
G["mamba config"]
A["COLABFOLD_READY check"]
B["pip install colabfold"]
C["TPU_NAME detection"]
D["JAX/dm-haiku setup"]

subgraph subGraph2 ["Optional Dependencies"]
    H
    J
    I
    K
    H --> J
    I --> K
end

subgraph subGraph1 ["Conda Environment"]
    E
    F
    G
    E --> F
    F --> G
end

subgraph subGraph0 ["Installation Logic"]
    A
    B
    C
    D
    A --> B
    B --> C
    C --> D
end
```

Both notebooks implement comprehensive dependency management [AlphaFold2.ipynb L145-L178](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L145-L178)

 using sentinel files to track installation state and conditional installation based on feature requirements like Amber relaxation or templates.

**Sources:** [AlphaFold2.ipynb L138-L186](https://github.com/sokrypton/ColabFold/blob/0c788a0e/AlphaFold2.ipynb#L138-L186)

 [batch/AlphaFold2_batch.ipynb L118-L163](https://github.com/sokrypton/ColabFold/blob/0c788a0e/batch/AlphaFold2_batch.ipynb#L118-L163)