---
title: "Compound Representation Learning"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.1-compound-representation-learning
---
# Compound Representation Learning

# Compound Representation Learning

> **Relevant source files**
> - [apps/pretrained\_compound/pretrain\_gnns/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1)
> - [apps/pretrained\_compound/pretrain\_gnns/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README_cn.md?plain=1)
> - [apps/pretrained\_compound/pretrain\_gnns/imgs/Evaluation\_results\.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/Evaluation_results.png)
> - [apps/pretrained\_compound/pretrain\_gnns/imgs/pregnn\.png](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/imgs/pregnn.png)
> - [tutorials/compound\_property\_prediction\_tutorial\.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb)
> - [tutorials/compound\_property\_prediction\_tutorial\_cn\.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial_cn.ipynb)

 This document covers PaddleHelix's compound representation learning system, which provides pretrained Graph Neural Network \(GNN\) models for molecular property prediction\. The system enables self\-supervised pretraining on large chemical datasets followed by fine\-tuning on specific downstream tasks\.

 For drug\-target interaction prediction, see [Drug\-Target Interaction](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction)\. For molecular generation capabilities, see [Molecular Generation](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.4-molecular-generation)\.

## System Architecture

 The compound representation learning system follows a two\-stage approach: pretraining on large unlabeled molecular datasets, followed by fine\-tuning on task\-specific labeled datasets\.

### Core Workflow

  Sources: [README\.md?plain=1 L40-L42](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L40-L42) [README\_cn\.md?plain=1 L40-L41](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README_cn.md?plain=1#L40-L41)

### Model Components

  Sources: [README\.md?plain=1 L134-L184](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L134-L184) [compound\_property\_prediction\_tutorial\.ipynb L57-L62](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L57-L62)

## Pretraining Strategies

### Node\-Level Pretraining

 The system implements attribute masking for self\-supervised pretraining at the node level using the `AttrmaskModel`\.

 **Attribute Masking Process:**

 1. Randomly masks atom types in molecular graphs
2. Trains the model to predict masked atom types based on neighboring structure
3. Uses ZINC15 dataset with 2 million unlabeled molecules

 Key parameters:

 - `mask_ratio`: Proportion of nodes to mask \(default: 0\.15\)
- Batch size: 256 for pretraining
- Learning rate: 0\.001

  Sources: [README\.md?plain=1 L196-L215](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L196-L215) [pretrain\_attrmask\.py L1-L100](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/pretrain_attrmask.py#L1-L100)

### Graph\-Level Supervised Pretraining

 Following node\-level pretraining, the system performs supervised pretraining on the entire graph using ChEMBL dataset\.

 **Supervised Pretraining Process:**

 1. Loads node\-level pretrained model
2. Trains on ChEMBL dataset with 456K molecules and 1310 biochemical assays
3. Performs multi\-task learning across different molecular properties

  Sources: [README\.md?plain=1 L216-L234](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L216-L234) [pretrain\_supervised\.py L1-L100](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/pretrain_supervised.py#L1-L100)

## GNN Architectures

### Available Models

| Model | Description | Key Parameters |
| --- | --- | --- |
| GIN | Graph Isomorphism Network | hidden\_size, embed\_dim, layer\_num |
| GAT | Graph Attention Network | hidden\_size, embed\_dim, layer\_num |
| GCN | Graph Convolutional Network | hidden\_size, embed\_dim, layer\_num |

### Model Configuration

 The system uses JSON configuration files to define model architectures:

 - `compound_encoder_config`: Defines the base GNN encoder
- `model_config`: Defines task\-specific model heads
- Common settings: `dropout_rate=0.2`, `layer_num=5`, `embed_dim=300`

  Sources: [README\.md?plain=1 L134-L184](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L134-L184) [compound\_property\_prediction\_tutorial\.ipynb L104-L111](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L104-L111)

## Downstream Applications

### Supported Datasets

 The system supports fine\-tuning on multiple molecular property prediction tasks:

| Dataset | Task | Molecules | Properties |
| --- | --- | --- | --- |
| BACE | β\-secretase inhibition | 1,522 | pIC50, binary class |
| BBBP | Blood\-brain barrier penetration | 2,000\+ | Binary permeability |
| ClinTox | Clinical trial toxicity | 1,491 | FDA approval, toxicity |
| HIV | HIV replication inhibition | 40,000\+ | Activity classification |
| Tox21 | Toxicity prediction | 8,000\+ | 12 toxicity assays |
| SIDER | Side effects | 1,427 | 27 organ classes |

### Fine\-tuning Process

  Sources: [README\.md?plain=1 L240-L267](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L240-L267) [compound\_property\_prediction\_tutorial\.ipynb L408-L419](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L408-L419)

## Usage Workflow

### Training Commands

 **Pretraining \(Attribute Masking\):**

  **Supervised Pretraining:**

  **Fine\-tuning:**

  Sources: [README\.md?plain=1 L89-L132](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/README.md?plain=1#L89-L132) [apps/pretrained\_compound/pretrain\_gnns/scripts/](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/scripts/)

### Data Processing Pipeline

  Sources: [pretrain\_gnn\_featurizer\.py L1-L100](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/pahelix/featurizers/pretrain_gnn_featurizer.py#L1-L100) [compound\_property\_prediction\_tutorial\.ipynb L185-L192](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L185-L192)

### Model Inference

 For inference on new SMILES strings, the system provides a streamlined interface:

  Sources: [compound\_property\_prediction\_tutorial\.ipynb L684-L696](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/compound_property_prediction_tutorial.ipynb#L684-L696) [finetune\.py L1-L50](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/pretrain_gnns/finetune.py#L1-L50)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.1-compound-representation-learning](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.1-compound-representation-learning) on DeepWiki*