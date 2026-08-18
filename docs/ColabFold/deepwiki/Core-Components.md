# Core Components

> **Relevant source files**
> * [colabfold/alphafold/extra_ptm.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/alphafold/extra_ptm.py)
> * [colabfold/batch.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py)
> * [colabfold/colabfold.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/colabfold.py)
> * [colabfold/pdb.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/pdb.py)
> * [colabfold/plot.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/plot.py)

This document covers the fundamental architectural components that form the backbone of the ColabFold system. These components handle the complete workflow from input sequence processing through MSA generation, model prediction, and result output.

For information about user interfaces and notebooks, see [Notebook Interfaces](/sokrypton/ColabFold/3.2-notebook-interfaces). For command-line tools, see [Command Line Interface](/sokrypton/ColabFold/4-command-line-interface). For advanced configuration and local execution, see [Advanced Usage](/sokrypton/ColabFold/5-advanced-usage).

## System Architecture Overview

The ColabFold core components form a modular pipeline architecture where each component has distinct responsibilities but works together to orchestrate protein structure prediction workflows.

### Core Component Relationships

```mermaid
flowchart TD

A["get_queries()"]
B["Input Validation"]
C["run_mmseqs2()"]
D["get_msa_and_templates()"]
E["MMseqs2 API Client"]
F["generate_input_feature()"]
G["build_monomer_feature()"]
H["build_multimer_feature()"]
I["predict_structure()"]
J["AlphaFold Models"]
K["model_runner.predict()"]
L["file_manager"]
M["relax_me()"]
N["Result Processing"]

A --> D
B --> D
D --> F
G --> I
H --> I
K --> L
I --> M

subgraph subGraph4 ["Output Management"]
    L
    M
    N
    M --> N
    L --> N
end

subgraph subGraph3 ["Prediction Engine"]
    I
    J
    K
    I --> J
    J --> K
end

subgraph subGraph2 ["Feature Engineering"]
    F
    G
    H
    F --> G
    F --> H
end

subgraph subGraph1 ["MSA Generation"]
    C
    D
    E
    D --> C
end

subgraph subGraph0 ["Input Layer"]
    A
    B
end
```

Sources: [colabfold/batch.py L700-L848](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L700-L848)

 [colabfold/batch.py L440-L698](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L440-L698)

 [colabfold/batch.py L942-L1044](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L942-L1044)

## Central Pipeline Controller

The `colabfold.batch` module serves as the central orchestrator for all protein structure prediction workflows. The main entry point coordinates the entire pipeline from input processing to final output.

### Batch Processing Pipeline

```mermaid
flowchart TD

A["run()"]
B["get_msa_and_templates()"]
C["generate_input_feature()"]
D["predict_structure()"]
E["file_manager"]
F["run_mmseqs2()"]
G["AlphaFold Models"]
H["relax_me()"]

B --> F
D --> G
D --> H

subgraph subGraph1 ["External APIs"]
    F
    G
    H
end

subgraph colabfold.batch ["colabfold.batch"]
    A
    B
    C
    D
    E
    A --> B
    B --> C
    C --> D
    D --> E
end
```

The batch processing system implements several key functions:

| Function | Purpose | Key Parameters |
| --- | --- | --- |
| `get_msa_and_templates()` | MSA generation and template search | `msa_mode`, `use_templates`, `pair_mode` |
| `generate_input_feature()` | Feature tensor preparation | `is_complex`, `model_type`, `max_seq` |
| `predict_structure()` | Model inference and ranking | `model_runner_and_params`, `num_relax` |
| `file_manager` | Result file organization | `prefix`, `result_dir` |

For details, see [Batch Processing System](/sokrypton/ColabFold/3.1-batch-processing-system).

Sources: [colabfold/batch.py L700-L848](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L700-L848)

 [colabfold/batch.py L942-L1044](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L942-L1044)

 [colabfold/batch.py L440-L698](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L440-L698)

 [colabfold/batch.py L423-L438](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L423-L438)

## MSA Generation and Search System

The MSA (Multiple Sequence Alignment) generation system is built around the `run_mmseqs2()` function, which interfaces with both remote API services and local MMseqs2 installations.

### MSA Generation Workflow

```mermaid
flowchart TD

A["run_mmseqs2()"]
B["submit()"]
C["status()"]
D["download()"]
E["ticket/msa"]
F["ticket/pair"]
G["result/download/{ID}"]
H["template/{TMPL_LINE}"]
I["a3m_lines extraction"]
J["template_paths setup"]
K["unserialize_msa()"]

B --> E
B --> F
C --> G
D --> H
D --> I

subgraph subGraph2 ["Output Processing"]
    I
    J
    K
    I --> J
    I --> K
end

subgraph subGraph1 ["API Endpoints"]
    E
    F
    G
    H
end

subgraph subGraph0 ["MSA Request Processing"]
    A
    B
    C
    D
    A --> B
    A --> C
    A --> D
end
```

The system supports multiple MSA modes:

* **Standard MSA**: `mmseqs2_uniref_env` - searches UniRef and environmental databases.
* **Pairing Mode**: For complex prediction with sequence pairing strategies like `greedy` or `complete`.
* **Template Search**: Retrieves structural templates when `use_templates` is enabled.

For details, see [MSA Generation and Search](/sokrypton/ColabFold/3.3-msa-generation-and-search).

Sources: [colabfold/colabfold.py L68-L331](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/colabfold.py#L68-L331)

 [colabfold/batch.py L700-L848](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L700-L848)

## Input Processing and Feature Engineering

The input processing system transforms raw sequences and MSAs into feature tensors compatible with AlphaFold models. This involves specialized functions depending on the prediction type (monomer vs multimer).

### Feature Generation Pipeline

```mermaid
flowchart TD

A["get_queries()"]
B["safe_filename()"]
C["pair_msa()"]
D["build_monomer_feature()"]
E["build_multimer_feature()"]
F["process_multimer_features()"]
G["mk_template()"]
H["mk_mock_template()"]
I["HhsearchHitFeaturizer"]

A --> D
A --> E
C --> D
C --> E
G --> D
H --> D

subgraph subGraph2 ["Template Processing"]
    G
    H
    I
    G --> I
end

subgraph subGraph1 ["Feature Building"]
    D
    E
    F
    E --> F
end

subgraph subGraph0 ["Input Processing"]
    A
    B
    C
end
```

Key feature engineering functions:

* `build_monomer_feature()`: Processes single chain features [colabfold/batch.py L850-L861](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L850-L861)
* `build_multimer_feature()`: Handles paired MSA and complex assembly [colabfold/batch.py L863-L868](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L863-L868)
* `mk_template()`: Featurizes HHsearch results for structural templates [colabfold/batch.py L145-L171](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L145-L171)

For details, see [Input Processing and Utilities](/sokrypton/ColabFold/3.4-input-processing-and-utilities).

Sources: [colabfold/batch.py L850-L861](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L850-L861)

 [colabfold/batch.py L863-L868](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L863-L868)

 [colabfold/batch.py L870-L931](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L870-L931)

 [colabfold/batch.py L109-L143](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L109-L143)

## Model Management and Prediction Engine

The prediction engine orchestrates AlphaFold model loading, inference, and result processing. It supports multiple model types and implements sophisticated ranking and relaxation workflows.

### Prediction Workflow

```mermaid
flowchart TD

A["load_models_and_params()"]
B["pad_input()"]
C["model_runner.process_features()"]
D["model_runner.predict()"]
E["callback()"]
F["protein.from_prediction()"]
G["relax_me()"]
H["Ranking by confidence"]
I["File output"]

C --> D
F --> G
F --> H

subgraph Post-Processing ["Post-Processing"]
    G
    H
    I
    G --> I
    H --> I
end

subgraph subGraph1 ["Prediction Loop"]
    D
    E
    F
    D --> E
    E --> F
end

subgraph subGraph0 ["Model Setup"]
    A
    B
    C
    A --> B
    B --> C
end
```

The prediction system handles multiple seeds, model ensembles (model_1 through model_5), and iterative recycling. It also integrates confidence scoring using pLDDT, pTM, and ipTM metrics [colabfold/alphafold/extra_ptm.py L42-L104](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/alphafold/extra_ptm.py#L42-L104)

For details, see [Model Management](/sokrypton/ColabFold/3.5-model-management).

Sources: [colabfold/batch.py L440-L698](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L440-L698)

 [colabfold/batch.py L389-L421](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L389-L421)

 [colabfold/alphafold/extra_ptm.py L42-L104](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/alphafold/extra_ptm.py#L42-L104)

## Visualization and Output System

The visualization system provides comprehensive analysis tools for structure prediction results, while the `file_manager` handles structured data serialization.

### Visualization and Output Schema

```mermaid
flowchart TD

A["set_tag()"]
B["get()"]
C["plot_msa_v2()"]
D["plot_predicted_alignment_error()"]
E["show_pdb()"]
F["{prefix}unrelaxed_rank*.pdb"]
G["{prefix}_PAE.png"]
H["{prefix}_scores.json"]

B --> F
B --> H
C --> G
D --> G
E --> F

subgraph subGraph2 ["Output Files"]
    F
    G
    H
end

subgraph Visualizers ["Visualizers"]
    C
    D
    E
end

subgraph file_manager ["file_manager"]
    A
    B
    A --> B
end
```

Key capabilities:

* **3D Structure Display**: Interactive `py3Dmol` viewers with confidence coloring [colabfold/pdb.py L1-L69](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/pdb.py#L1-L69)
* **Confidence Plots**: PAE (Predicted Aligned Error) heatmaps and pLDDT plots [colabfold/plot.py L4-L18](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/plot.py#L4-L18)
* **MSA Analysis**: Sequence coverage and identity visualization [colabfold/plot.py L20-L78](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/plot.py#L20-L78)

For details, see [Visualization and Output](/sokrypton/ColabFold/3.6-visualization-and-output).

Sources: [colabfold/batch.py L423-L438](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/batch.py#L423-L438)

 [colabfold/plot.py L20-L78](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/plot.py#L20-L78)

 [colabfold/pdb.py L1-L69](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/pdb.py#L1-L69)

 [colabfold/plot.py L4-L18](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/plot.py#L4-L18)