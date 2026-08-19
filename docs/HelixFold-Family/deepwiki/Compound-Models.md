# Compound Models

> **Relevant source files**
> * [apps/drug_target_interaction/graph_dta/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1)
> * [apps/drug_target_interaction/graph_dta/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README_cn.md?plain=1)
> * [apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py)
> * [apps/drug_target_interaction/graph_dta/scripts/train.sh](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/train.sh)
> * [apps/pretrained_compound/info_graph/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README.md?plain=1)
> * [apps/pretrained_compound/info_graph/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README_cn.md?plain=1)
> * [apps/pretrained_compound/info_graph/scripts/preprocess_data.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/scripts/preprocess_data.py)
> * [apps/pretrained_compound/info_graph/scripts/unsupervised_pretrain.sh](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/scripts/unsupervised_pretrain.sh)
> * [apps/pretrained_compound/info_graph/src/model.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py)
> * [apps/pretrained_compound/pretrain_gnns/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1)
> * [apps/pretrained_compound/pretrain_gnns/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README_cn.md?plain=1)
> * [apps/pretrained_compound/pretrain_gnns/imgs/Evaluation_results.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/Evaluation_results.png)
> * [apps/pretrained_compound/pretrain_gnns/imgs/pregnn.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/pregnn.png)
> * [tutorials/compound_property_prediction_tutorial.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb)
> * [tutorials/compound_property_prediction_tutorial_cn.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial_cn.ipynb)

This page documents the pretrained compound representation models available in PaddleHelix, including their architectures, training methodologies, and applications. These models enable learning molecular representations from graph neural networks for various drug discovery tasks.

For protein representation models, see [Protein Models](/PaddlePaddle/PaddleHelix/4.1-protein-models). For specific drug discovery applications using these models, see [Drug-Target Interaction](/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction) and [Drug-Drug Synergy](/PaddlePaddle/PaddleHelix/3.2.3-drug-drug-synergy).

## System Overview

PaddleHelix provides two main approaches for learning compound representations: **PretrainGNNs** and **InfoGraph**. Both systems use graph neural networks to encode molecular structures, but employ different pretraining strategies to learn generalizable molecular representations.

### Architecture Overview

```mermaid
flowchart TD

NODE_LEVEL["Node-level Pretraining<br>Attribute Masking"]
PRETRAIN_GNN["PretrainGNNModel"]
GIN["GIN"]
GAT["GAT"]
GCN["GCN"]
INFO_ENCODER["GINEncoder"]
ATTRMASK["AttrmaskModel"]
SUPERVISED["SupervisedModel"]
GRAPH_LEVEL["Graph-level Pretraining<br>Supervised Learning"]
INFO_MODEL["InfoGraph"]
MUTUAL_INFO["Mutual Information<br>Maximization"]
DOWNSTREAM["DownstreamModel"]
INFO_CRITERION["InfoGraphCriterion"]

subgraph subGraph4 ["Compound Models System"]
    PRETRAIN_GNN --> GIN
    PRETRAIN_GNN --> GAT
    PRETRAIN_GNN --> GCN
    INFO_ENCODER --> GIN
    ATTRMASK --> NODE_LEVEL
    SUPERVISED --> GRAPH_LEVEL
    INFO_MODEL --> MUTUAL_INFO
    NODE_LEVEL --> DOWNSTREAM
    GRAPH_LEVEL --> DOWNSTREAM
    MUTUAL_INFO --> INFO_CRITERION

subgraph subGraph3 ["Training Strategies"]
    NODE_LEVEL
    GRAPH_LEVEL
    MUTUAL_INFO
end

subgraph subGraph2 ["GNN Architectures"]
    GIN
    GAT
    GCN
end

subgraph subGraph1 ["InfoGraph Framework"]
    INFO_ENCODER
    INFO_MODEL
    INFO_CRITERION
end

subgraph subGraph0 ["PretrainGNNs Framework"]
    PRETRAIN_GNN
    ATTRMASK
    SUPERVISED
    DOWNSTREAM
end
end
```

Sources: [apps/pretrained_compound/pretrain_gnns/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1)

 [apps/pretrained_compound/info_graph/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README.md?plain=1)

## PretrainGNN Models

The PretrainGNN framework implements the strategies from "Strategies for Pre-training Graph Neural Networks" paper, providing both node-level and graph-level pretraining approaches for molecular representation learning.

### Core Components

| Component | Purpose | Implementation |
| --- | --- | --- |
| `PretrainGNNModel` | Base GNN encoder | [pahelix.model_zoo.pretrain_gnns_model.PretrainGNNModel](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/pahelix.model_zoo.pretrain_gnns_model.PretrainGNNModel) |
| `AttrmaskModel` | Node-level pretraining | [pahelix.model_zoo.pretrain_gnns_model.AttrmaskModel](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/pahelix.model_zoo.pretrain_gnns_model.AttrmaskModel) |
| `DownstreamModel` | Fine-tuning wrapper | [apps/pretrained_compound/pretrain_gnns/src/model.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/src/model.py) |

### Training Pipeline

```mermaid
flowchart TD

DOWNSTREAM_DATA["Downstream Datasets<br>BACE, BBBP, Tox21, etc."]
RAW_DATA["Raw Molecular Data<br>ZINC15, ChEMBL"]
ZINC_DATA["ZINC15 Dataset<br>2M unlabeled molecules"]
CHEMBL_DATA["ChEMBL Dataset<br>456K molecules, 1310 assays"]
ATTR_TRANSFORM["AttrmaskTransformFn"]
ATTR_MODEL["AttrmaskModel"]
ATTR_LOSS["Attribute Prediction Loss"]
SUP_TRANSFORM["SupervisedTransformFn"]
SUP_MODEL["SupervisedModel"]
SUP_LOSS["Multi-task Supervised Loss"]
PRETRAINED_ENCODER["Pretrained Compound Encoder"]
DOWN_MODEL["DownstreamModel"]
DOWN_TRANSFORM["DownstreamTransformFn"]
DOWN_LOSS["Task-specific Loss"]

subgraph subGraph3 ["PretrainGNN Training Pipeline"]
    RAW_DATA
    PRETRAINED_ENCODER
    RAW_DATA --> ZINC_DATA
    RAW_DATA --> CHEMBL_DATA
    RAW_DATA --> DOWNSTREAM_DATA
    ATTR_LOSS --> PRETRAINED_ENCODER
    SUP_LOSS --> PRETRAINED_ENCODER
    PRETRAINED_ENCODER --> DOWN_MODEL

subgraph Fine-tuning ["Fine-tuning"]
    DOWNSTREAM_DATA
    DOWN_MODEL
    DOWN_TRANSFORM
    DOWN_LOSS
    DOWNSTREAM_DATA --> DOWN_TRANSFORM
    DOWN_TRANSFORM --> DOWN_MODEL
    DOWN_MODEL --> DOWN_LOSS
end

subgraph subGraph1 ["Graph-level Pretraining"]
    CHEMBL_DATA
    SUP_TRANSFORM
    SUP_MODEL
    SUP_LOSS
    CHEMBL_DATA --> SUP_TRANSFORM
    SUP_TRANSFORM --> SUP_MODEL
    SUP_MODEL --> SUP_LOSS
end

subgraph subGraph0 ["Node-level Pretraining"]
    ZINC_DATA
    ATTR_TRANSFORM
    ATTR_MODEL
    ATTR_LOSS
    ZINC_DATA --> ATTR_TRANSFORM
    ATTR_TRANSFORM --> ATTR_MODEL
    ATTR_MODEL --> ATTR_LOSS
end
end
```

Sources: [apps/pretrained_compound/pretrain_gnns/README.md L61-L229](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L61-L229)

 [tutorials/compound_property_prediction_tutorial.ipynb L11-L14](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L11-L14)

### Pretraining Strategies

#### Node-level Pretraining (Attribute Masking)

The attribute masking strategy randomly masks node/edge attributes and trains the model to predict these masked attributes based on the graph structure.

**Configuration**: [apps/pretrained_compound/pretrain_gnns/model_configs/pre_Attrmask.json](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/model_configs/pre_Attrmask.json)

**Training Script**: [apps/pretrained_compound/pretrain_gnns/pretrain_attrmask.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/pretrain_attrmask.py)

```mermaid
flowchart TD

ORIG_MOL["Original Molecule"]
MASK_MOL["Masked Molecule<br>15% nodes masked"]
GNN_ENCODE["GNN Encoder"]
PREDICT["Attribute Prediction"]
LOSS["Prediction Loss"]

subgraph subGraph0 ["Attribute Masking Process"]
    ORIG_MOL
    MASK_MOL
    GNN_ENCODE
    PREDICT
    LOSS
    ORIG_MOL --> MASK_MOL
    MASK_MOL --> GNN_ENCODE
    GNN_ENCODE --> PREDICT
    PREDICT --> LOSS
end
```

#### Graph-level Pretraining (Supervised Learning)

Multi-task supervised pretraining uses the ChEMBL dataset to predict various biochemical assays, learning graph-level representations.

**Configuration**: [apps/pretrained_compound/pretrain_gnns/model_configs/pre_Supervised.json](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/model_configs/pre_Supervised.json)

**Training Script**: [apps/pretrained_compound/pretrain_gnns/pretrain_supervised.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/pretrain_supervised.py)

Sources: [apps/pretrained_compound/pretrain_gnns/README.md L208-L228](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L208-L228)

 [apps/pretrained_compound/pretrain_gnns/README_cn.md L181-L213](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README_cn.md?plain=1#L181-L213)

## InfoGraph Models

InfoGraph learns graph-level representations through maximizing mutual information between global graph representations and local substructure representations.

### Core Architecture

| Component | Purpose | Implementation |
| --- | --- | --- |
| `GINEncoder` | Graph encoding backbone | [apps/pretrained_compound/info_graph/src/model.py L31-L90](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py#L31-L90) |
| `InfoGraph` | Main model with MI maximization | [apps/pretrained_compound/info_graph/src/model.py L136-L161](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py#L136-L161) |
| `InfoGraphCriterion` | Mutual information loss | [apps/pretrained_compound/info_graph/src/model.py L163-L192](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py#L163-L192) |

### Training Process

```mermaid
flowchart TD

INPUT_GRAPH["Input Molecular Graph"]
GIN_ENC["GINEncoder"]
GLOBAL_REPR["Global Representation"]
PATCH_REPR["Patch Representations"]
FF_GLOBAL["Feedforward Network"]
FF_PATCH["Feedforward Network"]
MI_LOSS["Mutual Information Loss"]

subgraph subGraph0 ["InfoGraph Training"]
    INPUT_GRAPH
    GIN_ENC
    GLOBAL_REPR
    PATCH_REPR
    FF_GLOBAL
    FF_PATCH
    MI_LOSS
    INPUT_GRAPH --> GIN_ENC
    GIN_ENC --> GLOBAL_REPR
    GIN_ENC --> PATCH_REPR
    GLOBAL_REPR --> FF_GLOBAL
    PATCH_REPR --> FF_PATCH
    FF_GLOBAL --> MI_LOSS
    FF_PATCH --> MI_LOSS
end
```

Sources: [apps/pretrained_compound/info_graph/src/model.py L136-L192](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py#L136-L192)

 [apps/pretrained_compound/info_graph/README.md L14-L16](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README.md?plain=1#L14-L16)

## Model Architectures

PaddleHelix supports three main GNN architectures for compound encoding:

### Architecture Comparison

| Architecture | Description | Key Features | Configuration |
| --- | --- | --- | --- |
| **GIN** | Graph Isomorphism Network | Theoretically powerful for graph classification | [apps/pretrained_compound/pretrain_gnns/model_configs/pregnn_paper.json](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/model_configs/pregnn_paper.json) |
| **GAT** | Graph Attention Network | Attention-based neighbor aggregation | Edge-dependent attention weights |
| **GCN** | Graph Convolutional Network | Classic spectral-based convolution | Efficient message passing |

### Model Configuration

```mermaid
flowchart TD

ATOM_NAMES["atom_names: [atomic_num, chiral_tag]"]
CONFIG["Model Config JSON"]
EMBED_DIM["embed_dim: 300"]
HIDDEN_SIZE["hidden_size: varies"]
LAYER_NUM["layer_num: 5"]
GNN_TYPE["gnn_type: gin/gat/gcn"]
DROPOUT["dropout_rate: 0.5"]
NORM_TYPE["norm_type: batch_norm"]
READOUT["readout: mean"]
JK["JK: last"]
BOND_NAMES["bond_names: [bond_dir, bond_type]"]

subgraph subGraph3 ["GNN Model Configuration"]
    CONFIG
    CONFIG --> EMBED_DIM
    CONFIG --> HIDDEN_SIZE
    CONFIG --> LAYER_NUM
    CONFIG --> GNN_TYPE
    CONFIG --> DROPOUT
    CONFIG --> NORM_TYPE
    CONFIG --> READOUT
    CONFIG --> JK
    CONFIG --> ATOM_NAMES
    CONFIG --> BOND_NAMES

subgraph subGraph2 ["Feature Parameters"]
    ATOM_NAMES
    BOND_NAMES
end

subgraph subGraph1 ["Training Parameters"]
    DROPOUT
    NORM_TYPE
    READOUT
    JK
end

subgraph subGraph0 ["Architecture Parameters"]
    EMBED_DIM
    HIDDEN_SIZE
    LAYER_NUM
    GNN_TYPE
end
end
```

Sources: [apps/pretrained_compound/pretrain_gnns/model_configs/pregnn_paper.json](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/model_configs/pregnn_paper.json)

 [apps/pretrained_compound/pretrain_gnns/README.md L132-L177](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L132-L177)

## Training and Fine-tuning Pipeline

### Complete Training Workflow

```mermaid
flowchart TD

METRICS["ROC-AUC, Accuracy"]
ZINC_DOWNLOAD["wget zinc dataset"]
TRANSFORM["Data Transformation"]
DOWNSTREAM_DOWNLOAD["wget downstream datasets"]
ATTR_SCRIPT["pretrain_attrmask.py"]
SUP_SCRIPT["pretrain_supervised.py"]
SAVED_MODEL["Saved Compound Encoder"]
FINETUNE_SCRIPT["finetune.py"]
TASK_DATASETS["Task-specific Datasets<br>BACE, BBBP, Tox21, etc."]
FINETUNED_MODEL["Fine-tuned Model"]
RESULTS["Performance Results"]

subgraph subGraph4 ["End-to-End Training Pipeline"]
    TRANSFORM --> ATTR_SCRIPT
    TRANSFORM --> SUP_SCRIPT
    SAVED_MODEL --> FINETUNE_SCRIPT
    FINETUNED_MODEL --> METRICS

subgraph Evaluation ["Evaluation"]
    METRICS
    RESULTS
    METRICS --> RESULTS
end

subgraph subGraph2 ["Fine-tuning Phase"]
    FINETUNE_SCRIPT
    TASK_DATASETS
    FINETUNED_MODEL
    TASK_DATASETS --> FINETUNE_SCRIPT
    FINETUNE_SCRIPT --> FINETUNED_MODEL
end

subgraph subGraph1 ["Pretraining Phase"]
    ATTR_SCRIPT
    SUP_SCRIPT
    SAVED_MODEL
    ATTR_SCRIPT --> SAVED_MODEL
    SUP_SCRIPT --> SAVED_MODEL
end

subgraph subGraph0 ["Data Preparation"]
    ZINC_DOWNLOAD
    TRANSFORM
    DOWNSTREAM_DOWNLOAD
    ZINC_DOWNLOAD --> TRANSFORM
    DOWNSTREAM_DOWNLOAD --> TRANSFORM
end
end
```

### Training Scripts and Commands

**Pretraining Commands**:

```markdown
# Node-level pretrainingpython pretrain_attrmask.py --batch_size=256 --max_epoch=100 --data_path=../../../data/chem_dataset/zinc_standard_agent # Graph-level pretraining  python pretrain_supervised.py --batch_size=256 --max_epoch=100 --data_path=../../../data/chem_dataset/chembl_filtered
```

**Fine-tuning Command**:

```
python finetune.py --batch_size=32 --max_epoch=4 --dataset_name=tox21 --init_model=path/to/pretrained/model
```

Sources: [apps/pretrained_compound/pretrain_gnns/README.md L89-L132](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L89-L132)

 [tutorials/compound_property_prediction_tutorial.ipynb L195-L253](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L195-L253)

## Applications and Integration

### Downstream Task Integration

The pretrained compound models integrate with various drug discovery applications:

```mermaid
flowchart TD

DAVIS["Davis Dataset"]
PRETRAINED["Pretrained Compound Encoder"]
DTI["Drug-Target Interaction<br>GraphDTA, BatchDTA"]
DDS["Drug-Drug Synergy<br>RGCN, DTSyn"]
PROP_PRED["Property Prediction<br>ADMET, Toxicity"]
MOL_GEN["Molecular Generation<br>JT-VAE, seq-VAE"]
KIBA["KIBA Dataset"]
TOX21["Tox21 Dataset"]
BACE["BACE Dataset"]
BBBP["BBBP Dataset"]

subgraph subGraph2 ["Compound Model Applications"]
    PRETRAINED
    PRETRAINED --> DTI
    PRETRAINED --> DDS
    PRETRAINED --> PROP_PRED
    PRETRAINED --> MOL_GEN
    DTI --> DAVIS
    DTI --> KIBA
    PROP_PRED --> TOX21
    PROP_PRED --> BACE
    PROP_PRED --> BBBP

subgraph subGraph1 ["Evaluation Datasets"]
    DAVIS
    KIBA
    TOX21
    BACE
    BBBP
end

subgraph subGraph0 ["Drug Discovery Tasks"]
    DTI
    DDS
    PROP_PRED
    MOL_GEN
end
end
```

### Performance Results

Based on the evaluation results for molecular property prediction tasks:

| Dataset | Method | ROC-AUC | Task Type |
| --- | --- | --- | --- |
| Tox21 | PretrainGNN | 0.67+ | Toxicity Prediction |
| BACE | PretrainGNN | - | BACE-1 Inhibition |
| BBBP | PretrainGNN | - | Blood-Brain Barrier Permeability |

Sources: [apps/pretrained_compound/pretrain_gnns/README.md L259-L272](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L259-L272)

 [apps/drug_target_interaction/graph_dta/README.md L111-L128](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1#L111-L128)

## Dataset Support

### Pretraining Datasets

| Dataset | Purpose | Size | Description |
| --- | --- | --- | --- |
| **ZINC15** | Node-level pretraining | 2M molecules | Unlabeled molecular structures |
| **ChEMBL** | Graph-level pretraining | 456K molecules, 1310 assays | Biochemical activity data |

### Downstream Datasets

| Dataset | Task | Molecules | Description |
| --- | --- | --- | --- |
| **BACE** | Binary classification | 1,522 | BACE-1 inhibitor prediction |
| **BBBP** | Binary classification | 2,000+ | Blood-brain barrier permeability |
| **ClinTox** | Multi-task classification | 1,491 | Clinical trial toxicity |
| **HIV** | Binary classification | 40,000+ | HIV replication inhibition |
| **Tox21** | Multi-task classification | 8,000 | Toxicity across 12 targets |
| **SIDER** | Multi-task classification | 1,427 | Drug side effects |

### Data Processing Pipeline

```mermaid
flowchart TD

RAW_SMILES["Raw SMILES"]
RDKIT_MOL["RDKit Mol Object"]
GRAPH_DATA["Graph Data"]
FEATURES["Node/Edge Features"]
BATCH["Batched Graphs"]

subgraph subGraph0 ["Data Processing Pipeline"]
    RAW_SMILES
    RDKIT_MOL
    GRAPH_DATA
    FEATURES
    BATCH
    RAW_SMILES --> RDKIT_MOL
    RDKIT_MOL --> GRAPH_DATA
    GRAPH_DATA --> FEATURES
    FEATURES --> BATCH
end
```

**Key Processing Functions**:

* `AttrmaskTransformFn`: Node-level pretraining data transformation
* `DownstreamTransformFn`: Fine-tuning data transformation
* `AttrmaskCollateFn`: Batch collation for attribute masking
* `DownstreamCollateFn`: Batch collation for downstream tasks

Sources: [apps/pretrained_compound/pretrain_gnns/README.md L267-L375](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L267-L375)

 [pahelix.featurizers.pretrain_gnn_featurizer](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/pahelix.featurizers.pretrain_gnn_featurizer)