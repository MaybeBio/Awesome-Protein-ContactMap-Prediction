# Pretrained Models

> **Relevant source files**
> * [apps/pretrained_compound/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/README.md?plain=1)
> * [apps/pretrained_compound/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/README_cn.md?plain=1)
> * [apps/pretrained_compound/pretrain_gnns/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1)
> * [apps/pretrained_compound/pretrain_gnns/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README_cn.md?plain=1)
> * [apps/pretrained_compound/pretrain_gnns/imgs/Evaluation_results.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/Evaluation_results.png)
> * [apps/pretrained_compound/pretrain_gnns/imgs/pregnn.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/pregnn.png)
> * [apps/pretrained_protein/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/README.md?plain=1)
> * [apps/pretrained_protein/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/README_cn.md?plain=1)
> * [apps/pretrained_protein/tape/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1)
> * [apps/pretrained_protein/tape/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README_cn.md?plain=1)
> * [tutorials/compound_property_prediction_tutorial.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb)
> * [tutorials/compound_property_prediction_tutorial_cn.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial_cn.ipynb)

This document provides an overview of pretrained models available in PaddleHelix, covering both protein sequence models and compound graph neural network models. These pretrained models serve as foundation models that can be fine-tuned for downstream biological tasks, enabling transfer learning and improved performance on specialized tasks with limited data.

For detailed information about protein-specific models and tasks, see [Protein Models](/PaddlePaddle/PaddleHelix/4.1-protein-models). For compound representation learning and molecular property prediction, see [Compound Models](/PaddlePaddle/PaddleHelix/4.2-compound-models).

## Overview

PaddleHelix provides pretrained models for two primary biological domains: protein sequences and chemical compounds. Both domains follow a similar paradigm of large-scale unsupervised pretraining followed by task-specific fine-tuning.

### Pretrained Model Architecture

```mermaid
flowchart TD

PROTEIN_TASKS["Secondary Structure<br>Remote Homology<br>Fluorescence<br>Stability"]
TAPE["TAPE Framework"]
TRANSFORMER["Transformer Model"]
LSTM["LSTM Model"]
RESNET["ResNet Model"]
PFAM["Pfam Pretraining Dataset<br>30M sequences"]
PRETRAIN_GNN["PretrainGNNModel"]
ATTRMASK["AttrmaskModel"]
SUPERVISED["Supervised Pretraining"]
ZINC["ZINC15 Dataset<br>2M molecules"]
CHEMBL["ChEMBL Dataset<br>456K molecules"]
COMPOUND_TASKS["Molecular Properties<br>Drug-Target Interaction<br>Toxicity Prediction"]

subgraph subGraph3 ["Pretrained Model Framework"]
    TRANSFORMER --> PROTEIN_TASKS
    LSTM --> PROTEIN_TASKS
    RESNET --> PROTEIN_TASKS
    PRETRAIN_GNN --> COMPOUND_TASKS

subgraph subGraph2 ["Downstream Applications"]
    PROTEIN_TASKS
    COMPOUND_TASKS
end

subgraph subGraph1 ["Compound Domain"]
    PRETRAIN_GNN
    ATTRMASK
    SUPERVISED
    ZINC
    CHEMBL
    PRETRAIN_GNN --> ATTRMASK
    PRETRAIN_GNN --> SUPERVISED
    ZINC --> ATTRMASK
    CHEMBL --> SUPERVISED
end

subgraph subGraph0 ["Protein Domain"]
    TAPE
    TRANSFORMER
    LSTM
    RESNET
    PFAM
    TAPE --> TRANSFORMER
    TAPE --> LSTM
    TAPE --> RESNET
    PFAM --> TRANSFORMER
    PFAM --> LSTM
    PFAM --> RESNET
end
end
```

**Sources:** [apps/pretrained_protein/tape/README.md L1-L452](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L1-L452)

 [apps/pretrained_compound/pretrain_gnns/README.md L1-L549](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L1-L549)

### Available Pretrained Models

| Domain | Model Type | Architecture | Pretraining Dataset | Tasks |
| --- | --- | --- | --- | --- |
| Protein | TAPE Transformer | Transformer | Pfam (30M sequences) | Secondary structure, Remote homology, Fluorescence, Stability |
| Protein | TAPE LSTM | BiLSTM | Pfam (30M sequences) | Secondary structure, Remote homology, Fluorescence, Stability |
| Protein | TAPE ResNet | ResNet | Pfam (30M sequences) | Secondary structure, Remote homology, Fluorescence, Stability |
| Compound | PretrainGNNs | GIN/GAT/GCN | ZINC15 (2M molecules) | Molecular properties, Toxicity, Drug discovery |
| Compound | InfoGraph | GNN | Various datasets | Graph representation learning |

**Sources:** [apps/pretrained_protein/tape/README.md L300-L305](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L300-L305)

 [apps/pretrained_compound/pretrain_gnns/README.md L48-L51](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L48-L51)

## Code Architecture and Key Components

### Protein Models Code Structure

```mermaid
flowchart TD

PROTEIN_DATASET["ProteinDataset"]
FEATURIZER["ProteinFeaturizer"]
COLLATE_FN["CollateFn"]
TRAIN_PY["train.py<br>Training script"]
TAPE_MODEL["TAPEModel<br>Base model class"]
EVAL_PY["eval.py<br>Evaluation script"]
PREDICT_PY["predict.py<br>Inference script"]
TRANSFORMER_CONFIG["transformer_config.json"]
TRANSFORMER_MODEL["TransformerModel"]
LSTM_CONFIG["lstm_config.json"]
LSTM_MODEL["LSTMModel"]
RESNET_CONFIG["resnet_config.json"]
RESNET_MODEL["ResNetModel"]

subgraph subGraph3 ["TAPE Framework Components"]
    TRAIN_PY
    EVAL_PY
    PREDICT_PY
    TRAIN_PY --> TAPE_MODEL
    EVAL_PY --> TAPE_MODEL
    PREDICT_PY --> TAPE_MODEL
    TRANSFORMER_CONFIG --> TRANSFORMER_MODEL
    LSTM_CONFIG --> LSTM_MODEL
    RESNET_CONFIG --> RESNET_MODEL

subgraph subGraph1 ["Core Classes"]
    TAPE_MODEL
    TRANSFORMER_MODEL
    LSTM_MODEL
    RESNET_MODEL
    TAPE_MODEL --> TRANSFORMER_MODEL
    TAPE_MODEL --> LSTM_MODEL
    TAPE_MODEL --> RESNET_MODEL
end

subgraph subGraph0 ["Model Configurations"]
    TRANSFORMER_CONFIG
    LSTM_CONFIG
    RESNET_CONFIG
end

subgraph subGraph2 ["Data Processing"]
    PROTEIN_DATASET
    FEATURIZER
    COLLATE_FN
    PROTEIN_DATASET --> FEATURIZER
    FEATURIZER --> COLLATE_FN
end
end
```

**Sources:** [apps/pretrained_protein/tape/README.md L47-L89](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L47-L89)

 [apps/pretrained_protein/tape/README.md L256-L290](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L256-L290)

### Compound Models Code Structure

```mermaid
flowchart TD

ATTRMASK_TRANSFORM["AttrmaskTransformFn"]
ATTRMASK_COLLATE["AttrmaskCollateFn"]
DOWNSTREAM_TRANSFORM["DownstreamTransformFn"]
DOWNSTREAM_COLLATE["DownstreamCollateFn"]
PRETRAIN_ATTR["pretrain_attrmask.py<br>Node-level pretraining"]
ATTRMASK_MODEL["AttrmaskModel<br>Attribute masking"]
PRETRAIN_SUP["pretrain_supervised.py<br>Graph-level pretraining"]
PRETRAIN_GNN_MODEL["PretrainGNNModel<br>Base GNN encoder"]
FINETUNE_PY["finetune.py<br>Downstream fine-tuning"]
DOWNSTREAM_MODEL["DownstreamModel<br>Task-specific head"]
GIN["GIN<br>Graph Isomorphism Network"]
GAT["GAT<br>Graph Attention Network"]
GCN["GCN<br>Graph Convolutional Network"]

subgraph subGraph4 ["PretrainGNNs Framework"]
    PRETRAIN_ATTR --> ATTRMASK_MODEL
    PRETRAIN_SUP --> PRETRAIN_GNN_MODEL
    FINETUNE_PY --> DOWNSTREAM_MODEL
    PRETRAIN_GNN_MODEL --> GIN
    PRETRAIN_GNN_MODEL --> GAT
    PRETRAIN_GNN_MODEL --> GCN

subgraph subGraph2 ["GNN Architectures"]
    GIN
    GAT
    GCN
end

subgraph subGraph1 ["Core Model Classes"]
    ATTRMASK_MODEL
    PRETRAIN_GNN_MODEL
    DOWNSTREAM_MODEL
    ATTRMASK_MODEL --> PRETRAIN_GNN_MODEL
    DOWNSTREAM_MODEL --> PRETRAIN_GNN_MODEL
end

subgraph subGraph0 ["Training Scripts"]
    PRETRAIN_ATTR
    PRETRAIN_SUP
    FINETUNE_PY
end

subgraph subGraph3 ["Data Processing"]
    ATTRMASK_TRANSFORM
    ATTRMASK_COLLATE
    DOWNSTREAM_TRANSFORM
    DOWNSTREAM_COLLATE
    ATTRMASK_TRANSFORM --> ATTRMASK_COLLATE
    DOWNSTREAM_TRANSFORM --> DOWNSTREAM_COLLATE
end
end
```

**Sources:** [apps/pretrained_compound/pretrain_gnns/README.md L67-L131](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L67-L131)

 [tutorials/compound_property_prediction_tutorial.ipynb L57-L61](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L57-L61)

## Pretraining and Fine-tuning Workflow

### General Training Pipeline

```mermaid
flowchart TD

RAW_DATA["Raw Biological Data<br>SMILES, Sequences"]
PREPROCESS["Data Preprocessing<br>Featurizers"]
PRETRAIN["Unsupervised Pretraining<br>Self-supervised tasks"]
PRETRAIN_MODEL["Pretrained Model<br>.pdparams files"]
DOWNSTREAM_DATA["Downstream Task Data<br>Labeled datasets"]
TASK_HEAD["Task-specific Head<br>Classification/Regression"]
FINETUNE["Supervised Fine-tuning<br>End-to-end training"]
FINAL_MODEL["Fine-tuned Model<br>Task-ready"]
NEW_INPUT["New Input<br>Query molecules/proteins"]
PREDICTION["Model Prediction<br>Properties/Functions"]

PRETRAIN_MODEL --> TASK_HEAD
FINAL_MODEL --> NEW_INPUT

subgraph subGraph2 ["Stage 3: Inference"]
    NEW_INPUT
    PREDICTION
    NEW_INPUT --> PREDICTION
end

subgraph subGraph1 ["Stage 2: Fine-tuning"]
    DOWNSTREAM_DATA
    TASK_HEAD
    FINETUNE
    FINAL_MODEL
    DOWNSTREAM_DATA --> TASK_HEAD
    TASK_HEAD --> FINETUNE
    FINETUNE --> FINAL_MODEL
end

subgraph subGraph0 ["Stage 1: Pretraining"]
    RAW_DATA
    PREPROCESS
    PRETRAIN
    PRETRAIN_MODEL
    RAW_DATA --> PREPROCESS
    PREPROCESS --> PRETRAIN
    PRETRAIN --> PRETRAIN_MODEL
end
```

**Sources:** [tutorials/compound_property_prediction_tutorial.ipynb L11-L261](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L11-L261)

 [apps/pretrained_protein/tape/README.md L245-L253](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L245-L253)

### Compound Model Training Example

The compound models follow a two-stage pretraining approach:

1. **Node-level Pretraining** using `AttrmaskModel`: Randomly masks atom types and predicts them based on graph structure
2. **Graph-level Pretraining**: Multi-task supervised learning on molecular properties

Key training components:

* `PretrainGNNModel` - Base GNN encoder supporting GIN, GAT, GCN architectures
* `AttrmaskTransformFn` - Transforms SMILES to graph features with masking
* `AttrmaskCollateFn` - Batches graphs for attribute masking tasks

**Sources:** [apps/pretrained_compound/pretrain_gnns/README.md L189-L236](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L189-L236)

 [tutorials/compound_property_prediction_tutorial.ipynb L104-L110](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L104-L110)

### Protein Model Training Example

The protein models use the TAPE framework:

1. **Pretraining** on Pfam dataset (30M protein sequences) with masked language modeling
2. **Fine-tuning** on specific tasks like secondary structure prediction

Key training components:

* Model configuration files (e.g., `transformer_config.json`)
* Training scripts supporting multi-GPU distributed training
* Task-specific evaluation metrics

**Sources:** [apps/pretrained_protein/tape/README.md L143-L290](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L143-L290)

## Model Access and Usage

### Available Pretrained Model Downloads

| Model | Download Link | Description |
| --- | --- | --- |
| TAPE Transformer | `baidu-nlp.bj.bcebos.com/.../tape_transformer_pretrain.pdparams` | Transformer pretrained on Pfam |
| TAPE LSTM | `baidu-nlp.bj.bcebos.com/.../tape_lstm_pretrain.pdparams` | BiLSTM pretrained on Pfam |
| TAPE ResNet | `baidu-nlp.bj.bcebos.com/.../tape_resnet_pretrain.pdparam` | ResNet pretrained on Pfam |
| PretrainGNNs | `baidu-nlp.bj.bcebos.com/.../pregnn-attrmask-supervised.zip` | GNN pretrained with attribute masking + supervised |

**Sources:** [apps/pretrained_protein/tape/README.md L301-L304](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L301-L304)

 [apps/pretrained_compound/pretrain_gnns/README.md L48-L50](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L48-L50)

### Loading and Using Models

Both protein and compound models follow similar loading patterns:

1. **Load model configuration** from JSON files
2. **Initialize model architecture** with configuration parameters
3. **Load pretrained weights** using `paddle.load()`
4. **Fine-tune or perform inference** on downstream tasks

The models integrate with PaddlePaddle's standard training and inference APIs, making them easy to incorporate into custom workflows.

**Sources:** [tutorials/compound_property_prediction_tutorial.ipynb L104-L110](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L104-L110)

 [tutorials/compound_property_prediction_tutorial.ipynb L346-L371](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L346-L371)