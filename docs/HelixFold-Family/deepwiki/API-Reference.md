# API Reference

> **Relevant source files**
> * [docs/api_doc/datasets.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst)
> * [docs/api_doc/featurizers.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst)
> * [docs/api_doc/model_zoo.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst)
> * [docs/api_doc/networks.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst)
> * [docs/api_doc/utils.rst](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst)
> * [docs/conf.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py)
> * [docs/requirements.txt](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/requirements.txt)

This page provides comprehensive documentation of the PaddleHelix Python API, automatically generated from source code docstrings using the Sphinx documentation system. The API reference covers all public classes, functions, and modules available for developers building bio-computing applications with PaddleHelix.

For general usage guides and tutorials, see [Getting Started](/PaddlePaddle/PaddleHelix/2-getting-started). For specific application domain guides, see [Core Applications](/PaddlePaddle/PaddleHelix/3-core-applications). For information about pretrained models, see [Pretrained Models](/PaddlePaddle/PaddleHelix/4-pretrained-models).

## API Documentation System

PaddleHelix uses Sphinx with autodoc extensions to automatically generate API documentation from Python docstrings. The documentation system is configured to handle the complex dependencies typical in bio-computing environments through mock imports and specialized parsing.

```mermaid
flowchart TD

SOURCE["Python Source Files<br>pahelix/*.py"]
DOCSTRINGS["Google/NumPy<br>Docstrings"]
SPHINX["Sphinx autodoc<br>conf.py"]
RST["RST Documentation<br>docs/api_doc/*.rst"]
HTML["Generated HTML<br>API Reference"]
CONF["docs/conf.py"]
MOCK["Mock Dependencies<br>paddle, pgl, rdkit"]
NAPOLEON["Napoleon Extension<br>Docstring Parsing"]
RTD["ReadTheDocs Theme"]

CONF --> SPHINX
MOCK --> SPHINX
NAPOLEON --> SPHINX
RTD --> HTML

subgraph Configuration ["Configuration"]
    CONF
    MOCK
    NAPOLEON
    RTD
end

subgraph subGraph0 ["Documentation Generation"]
    SOURCE
    DOCSTRINGS
    SPHINX
    RST
    HTML
    SOURCE --> DOCSTRINGS
    DOCSTRINGS --> SPHINX
    SPHINX --> RST
    RST --> HTML
end
```

*Sources: [docs/conf.py L1-L112](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py#L1-L112)

 [docs/requirements.txt L1-L6](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/requirements.txt#L1-L6)*

## API Module Structure

The PaddleHelix API is organized into five primary modules, each serving distinct functional areas of bio-computing applications:

| Module | Purpose | Key Components |
| --- | --- | --- |
| `pahelix.model_zoo` | Pretrained and task-specific models | `PretrainGNNModel`, `ProteinModel`, `VAE` |
| `pahelix.networks` | Neural network building blocks | `GIN`, `MLP`, `TransformerEncoder` |
| `pahelix.datasets` | Dataset loaders and management | `InMemoryDataset`, dataset loaders |
| `pahelix.featurizers` | Data preprocessing and feature extraction | `AttrmaskTransformFn`, `DDiFeaturizer` |
| `pahelix.utils` | Utilities and helper functions | `CompoundKit`, `ProteinTokenizer`, splitters |

```mermaid
flowchart TD

PRETRAIN_GNN["PretrainGNNModel"]
MODEL_ZOO["pahelix.model_zoo<br>Pretrained Models"]
PROTEIN_MODEL["ProteinModel"]
VAE_MODEL["VAE"]
NETWORKS["pahelix.networks<br>Neural Network Blocks"]
GIN_LAYER["GIN"]
MLP_LAYER["MLP"]
TRANSFORMER["TransformerEncoder"]
DATASETS["pahelix.datasets<br>Dataset Management"]
INMEMORY["InMemoryDataset"]
UTILS["pahelix.utils<br>Utility Functions"]
COMPOUND_KIT["CompoundKit"]
FEATURIZERS["pahelix.featurizers<br>Data Processing"]

subgraph subGraph1 ["pahelix API Structure"]
    MODEL_ZOO
    NETWORKS
    DATASETS
    UTILS
    FEATURIZERS
    MODEL_ZOO --> PRETRAIN_GNN
    MODEL_ZOO --> PROTEIN_MODEL
    MODEL_ZOO --> VAE_MODEL
    NETWORKS --> GIN_LAYER
    NETWORKS --> MLP_LAYER
    NETWORKS --> TRANSFORMER
    DATASETS --> INMEMORY
    UTILS --> COMPOUND_KIT

subgraph subGraph0 ["Core Classes"]
    PRETRAIN_GNN
    PROTEIN_MODEL
    VAE_MODEL
    GIN_LAYER
    MLP_LAYER
    TRANSFORMER
    INMEMORY
    COMPOUND_KIT
end
end
```

*Sources: [docs/api_doc/model_zoo.rst L1-L60](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L1-L60)

 [docs/api_doc/networks.rst L1-L75](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst#L1-L75)

 [docs/api_doc/datasets.rst L1-L147](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst#L1-L147)

 [docs/api_doc/featurizers.rst L1-L46](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L1-L46)

 [docs/api_doc/utils.rst L1-L72](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst#L1-L72)*

## Model Zoo API

The `pahelix.model_zoo` module provides pretrained models and task-specific architectures for bio-computing applications.

### Graph Neural Network Models

The pretrained GNN models support molecular property prediction and compound representation learning:

```mermaid
flowchart TD

BASE["PretrainGNNModel<br>Base Class"]
ATTRMASK["AttrmaskModel<br>Attribute Masking"]
SUPERVISED["SupervisedModel<br>Supervised Learning"]
ENCODER["GNN Encoder<br>GIN/GraphSAINT"]
PREDICTOR["Task Predictor<br>MLP Heads"]
POOLING["Graph Pooling<br>Mean/Max/Attention"]

ATTRMASK --> ENCODER
SUPERVISED --> ENCODER

subgraph subGraph1 ["Model Components"]
    ENCODER
    PREDICTOR
    POOLING
    ENCODER --> POOLING
    POOLING --> PREDICTOR
end

subgraph subGraph0 ["PretrainGNNModel Hierarchy"]
    BASE
    ATTRMASK
    SUPERVISED
    BASE --> ATTRMASK
    BASE --> SUPERVISED
end
```

Key classes include:

* `PretrainGNNModel`: Base class for graph neural network models
* `AttrmaskModel`: Self-supervised pretraining with attribute masking
* `SupervisedModel`: Supervised learning for molecular property prediction

### Protein Sequence Models

The protein modeling framework provides various encoder architectures:

```mermaid
flowchart TD

PRETRAIN_TASK["PretrainTaskModel"]
INPUT["Protein Sequence<br>Amino Acid Tokens"]
LSTM["LstmEncoderModel<br>Sequential Processing"]
RESNET["ResnetEncoderModel<br>Residual Blocks"]
TRANSFORMER["TransformerEncoderModel<br>Self-Attention"]
ENCODER_BASE["ProteinEncoderModel<br>Base Encoder"]
PROTEIN_MODEL["ProteinModel<br>Task-Specific Head"]
SEQ_CLASS["SeqClassificationTaskModel"]
CLASSIFICATION["ClassificationTaskModel"]
REGRESSION["RegressionTaskModel"]

subgraph subGraph2 ["Protein Model Architecture"]
    INPUT
    ENCODER_BASE
    PROTEIN_MODEL
    INPUT --> LSTM
    INPUT --> RESNET
    INPUT --> TRANSFORMER
    LSTM --> ENCODER_BASE
    RESNET --> ENCODER_BASE
    TRANSFORMER --> ENCODER_BASE
    ENCODER_BASE --> PROTEIN_MODEL
    PROTEIN_MODEL --> PRETRAIN_TASK
    PROTEIN_MODEL --> SEQ_CLASS
    PROTEIN_MODEL --> CLASSIFICATION
    PROTEIN_MODEL --> REGRESSION

subgraph subGraph1 ["Task Models"]
    PRETRAIN_TASK
    SEQ_CLASS
    CLASSIFICATION
    REGRESSION
end

subgraph subGraph0 ["Encoder Options"]
    LSTM
    RESNET
    TRANSFORMER
end
end
```

*Sources: [docs/api_doc/model_zoo.rst L7-L48](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/model_zoo.rst#L7-L48)*

## Networks API

The `pahelix.networks` module provides fundamental neural network building blocks used across PaddleHelix models.

### Core Building Blocks

| Component | Class | Purpose |
| --- | --- | --- |
| Activation | `Activation` | Configurable activation functions |
| Multi-layer Perceptron | `MLP` | Fully connected layers with dropout |
| Graph Convolution | `GIN` | Graph Isomorphism Network layers |
| Attention | `multi_head_attention` | Multi-head self-attention mechanism |
| Normalization | `GraphNorm` | Graph-specific normalization |

### Specialized Encoders

```mermaid
flowchart TD

ATTENTION["multi_head_attention<br>Scaled Dot-Product"]
SEQUENCE["Sequence Input<br>Tokens/Embeddings"]
LSTM_ENC["lstm_encoder<br>Bidirectional LSTM"]
RESNET_ENC["resnet_encoder<br>Residual Connections"]
TRANSFORMER_ENC["transformer_encoder<br>Self-Attention Layers"]
FFN["positionwise_feed_forward<br>Position-wise MLP"]
LAYER_NORM["pre_post_process_layer<br>Layer Normalization"]
OUTPUT["Encoded Representations"]

subgraph subGraph2 ["Encoder Architecture"]
    SEQUENCE
    OUTPUT
    SEQUENCE --> LSTM_ENC
    SEQUENCE --> RESNET_ENC
    SEQUENCE --> TRANSFORMER_ENC
    TRANSFORMER_ENC --> ATTENTION
    TRANSFORMER_ENC --> FFN
    LSTM_ENC --> OUTPUT
    RESNET_ENC --> OUTPUT
    TRANSFORMER_ENC --> OUTPUT

subgraph subGraph1 ["Supporting Blocks"]
    ATTENTION
    FFN
    LAYER_NORM
    ATTENTION --> LAYER_NORM
    FFN --> LAYER_NORM
end

subgraph subGraph0 ["Encoder Types"]
    LSTM_ENC
    RESNET_ENC
    TRANSFORMER_ENC
end
end
```

*Sources: [docs/api_doc/networks.rst L1-L75](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/networks.rst#L1-L75)*

## Datasets API

The `pahelix.datasets` module provides standardized dataset loaders for common bio-computing benchmarks and the `InMemoryDataset` framework for custom data handling.

### Dataset Categories

```mermaid
flowchart TD

CHEMBL["ChEMBL Filtered<br>load_chembl_filtered_dataset"]
BACE["BACE Dataset<br>load_bace_dataset"]
INMEMORY_DS["InMemoryDataset<br>Base Framework"]
DAVIS["DAVIS Dataset<br>load_davis_dataset"]
DDI["DDI Dataset<br>load_ddi_dataset"]
ZINC["ZINC Dataset<br>load_zinc_dataset"]
KIBA["KIBA Dataset<br>load_kiba_dataset"]
DTI["DTI Dataset<br>load_dti_dataset"]
BBBP["BBBP Dataset<br>load_bbbp_dataset"]
ESOL["ESOL Dataset<br>load_esol_dataset"]
HIV["HIV Dataset<br>load_hiv_dataset"]
TOX21["Tox21 Dataset<br>load_tox21_dataset"]

subgraph subGraph4 ["Dataset Categories"]
    INMEMORY_DS
    BACE --> INMEMORY_DS
    DAVIS --> INMEMORY_DS
    DDI --> INMEMORY_DS
    CHEMBL --> INMEMORY_DS

subgraph subGraph3 ["Large Scale"]
    CHEMBL
    ZINC
end

subgraph subGraph2 ["Drug-Drug Interaction"]
    DDI
end

subgraph subGraph1 ["Drug-Target Interaction"]
    DAVIS
    KIBA
    DTI
end

subgraph subGraph0 ["Molecular Property"]
    BACE
    BBBP
    ESOL
    HIV
    TOX21
end
end
```

### Dataset Loaders

All dataset loaders follow a consistent pattern and return data compatible with the `InMemoryDataset` framework:

* **Function naming**: `load_{dataset_name}_dataset()`
* **Task names**: `get_default_{dataset_name}_task_names()`
* **Data ranges**: `get_default_{dataset_name}_range()` (for regression tasks)

*Sources: [docs/api_doc/datasets.rst L1-L147](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/datasets.rst#L1-L147)*

## Featurizers API

The `pahelix.featurizers` module handles data preprocessing and feature extraction for different input modalities.

### Featurizer Types

```mermaid
flowchart TD

DDI_FEATURIZER["DDiFeaturizer<br>Drug-Drug Interactions"]
RAW_DATA["Raw Data<br>SMILES, Sequences"]
ATTRMASK_TRANSFORM["AttrmaskTransformFn<br>Masking for Pretraining"]
SUPERVISED_TRANSFORM["SupervisedTransformFn<br>Supervised Learning"]
ATTRMASK_COLLATE["AttrmaskCollateFn<br>Batch Processing"]
SUPERVISED_COLLATE["SupervisedCollateFn<br>Batch Processing"]
PROCESSED_DATA["Processed Features<br>Tensors, Graphs"]
HET_GNN["het_gnn_featurizer<br>Heterogeneous Graphs"]

subgraph subGraph3 ["Featurizer Pipeline"]
    RAW_DATA
    PROCESSED_DATA
    RAW_DATA --> ATTRMASK_TRANSFORM
    RAW_DATA --> SUPERVISED_TRANSFORM
    RAW_DATA --> DDI_FEATURIZER
    ATTRMASK_TRANSFORM --> ATTRMASK_COLLATE
    SUPERVISED_TRANSFORM --> SUPERVISED_COLLATE
    ATTRMASK_COLLATE --> PROCESSED_DATA
    SUPERVISED_COLLATE --> PROCESSED_DATA
    DDI_FEATURIZER --> PROCESSED_DATA

subgraph subGraph2 ["Specialized Featurizers"]
    DDI_FEATURIZER
    HET_GNN
end

subgraph subGraph1 ["Collate Functions"]
    ATTRMASK_COLLATE
    SUPERVISED_COLLATE
end

subgraph subGraph0 ["Transform Functions"]
    ATTRMASK_TRANSFORM
    SUPERVISED_TRANSFORM
end
end
```

*Sources: [docs/api_doc/featurizers.rst L1-L46](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/featurizers.rst#L1-L46)*

## Utils API

The `pahelix.utils` module provides essential utility functions for data processing, molecular manipulation, and model training support.

### Utility Categories

| Category | Module | Key Functions |
| --- | --- | --- |
| Basic Utilities | `basic_utils` | `mp_pool_map`, `load_json_config` |
| Compound Processing | `compound_tools` | `CompoundKit`, `Compound3DKit`, `mol_to_graph_data` |
| Data Management | `data_utils` | `save_data_list_to_npz`, `load_npz_to_data_list` |
| Protein Processing | `protein_tools` | `ProteinTokenizer` |
| Data Splitting | `splitters` | `RandomSplitter`, `ScaffoldSplitter` |
| Language Models | `language_model_tools` | `apply_bert_mask` |

### Compound Tools

The `CompoundKit` and related functions provide comprehensive molecular processing capabilities:

```mermaid
flowchart TD

BASIC_GRAPH["mol_to_graph_data<br>Basic Graph"]
SMILES["SMILES String<br>Chemical Notation"]
VALIDITY["check_smiles_validity<br>Validate SMILES"]
STANDARDIZE["create_standardized_mol_id<br>Canonical Form"]
LARGEST["get_largest_mol<br>Primary Component"]
COMPOUND_KIT["CompoundKit<br>2D Molecular Features"]
COMPOUND_3D["Compound3DKit<br>3D Conformations"]
CHARGES["get_gasteiger_partial_charges<br>Atomic Charges"]
MD_GRAPH["mol_to_md_graph_data<br>Multi-directional"]
POLAR_GRAPH["mol_to_polar_graph_data<br>Polar Coordinates"]
SUPER_GRAPH["mol_to_superedge_graph_data<br>Super Edges"]
GRAPH_DATA["Graph Data<br>Ready for GNNs"]

subgraph subGraph3 ["Compound Processing Pipeline"]
    SMILES
    GRAPH_DATA
    SMILES --> VALIDITY
    LARGEST --> COMPOUND_KIT
    LARGEST --> COMPOUND_3D
    LARGEST --> CHARGES
    COMPOUND_KIT --> BASIC_GRAPH
    COMPOUND_KIT --> MD_GRAPH
    COMPOUND_KIT --> POLAR_GRAPH
    COMPOUND_KIT --> SUPER_GRAPH
    BASIC_GRAPH --> GRAPH_DATA
    MD_GRAPH --> GRAPH_DATA
    POLAR_GRAPH --> GRAPH_DATA
    SUPER_GRAPH --> GRAPH_DATA

subgraph subGraph2 ["Graph Conversion"]
    BASIC_GRAPH
    MD_GRAPH
    POLAR_GRAPH
    SUPER_GRAPH
end

subgraph subGraph1 ["Feature Extraction"]
    COMPOUND_KIT
    COMPOUND_3D
    CHARGES
end

subgraph subGraph0 ["Validation & Standardization"]
    VALIDITY
    STANDARDIZE
    LARGEST
    VALIDITY --> STANDARDIZE
    STANDARDIZE --> LARGEST
end
end
```

*Sources: [docs/api_doc/utils.rst L1-L72](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/api_doc/utils.rst#L1-L72)*

## Documentation Configuration

The API documentation is generated using Sphinx with the following key configuration settings:

* **Extensions**: `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `sphinx.ext.viewcode`
* **Docstring Formats**: Google and NumPy docstring conventions
* **Mock Dependencies**: Automatic mocking of `paddle`, `pgl`, `rdkit`, and scientific packages
* **Theme**: ReadTheDocs theme with navigation depth of 5 levels
* **Version**: 1.0.0

The configuration handles the complex dependency requirements of bio-computing libraries through comprehensive mock imports, allowing documentation generation without requiring full installation of all dependencies.

*Sources: [docs/conf.py L1-L112](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/conf.py#L1-L112)

 [docs/requirements.txt L1-L6](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/docs/requirements.txt#L1-L6)*