---
title: "Drug-Drug Synergy"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.3-drug-drug-synergy
---
# Drug\-Drug Synergy

# Drug\-Drug Synergy

> **Relevant source files**
> - [apps/drug\_drug\_synergy/DTSyn/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/README.md?plain=1)
> - [apps/drug\_drug\_synergy/DTSyn/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/README_cn.md?plain=1)
> - [apps/drug\_drug\_synergy/DTSyn/\_\_init\_\_\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/__init__.py)
> - [apps/drug\_drug\_synergy/DTSyn/main\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/main.py)
> - [apps/drug\_drug\_synergy/DTSyn/tsnet\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/tsnet.py)
> - [apps/drug\_drug\_synergy/DTSyn/utils\_no\_de\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/utils_no_de.py)
> - [apps/drug\_drug\_synergy/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/README.md?plain=1)
> - [apps/drug\_drug\_synergy/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/README_cn.md?plain=1)
> - [apps/drug\_drug\_synergy/RGCN/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/README.md?plain=1)
> - [apps/drug\_drug\_synergy/RGCN/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/README_cn.md?plain=1)
> - [apps/drug\_drug\_synergy/RGCN/R\_model\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/R_model.py)
> - [apps/drug\_drug\_synergy/RGCN/graphsage\_sampling\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/graphsage_sampling.py)
> - [apps/drug\_drug\_synergy/RGCN/train\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/train.py)

 This document covers the drug\-drug synergy prediction capabilities within PaddleHelix, which provide computational methods for identifying synergistic drug combinations\. The system implements two distinct approaches: RGCN \(Relational Graph Convolutional Network\) and DTSyn \(Dual\-Transformer neural network\) for predicting drug combination efficacy\.

 For drug\-target interaction prediction, see [Drug\-Target Interaction](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction)\. For compound representation learning, see [Compound Representation Learning](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.1-compound-representation-learning)\.

## System Architecture Overview

 The drug\-drug synergy prediction system consists of two independent yet complementary approaches, each with distinct model architectures and data processing pipelines\.

  **Sources:** [README\.md?plain=1 L1-L9](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/README.md?plain=1#L1-L9) [train\.py L1-L168](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/train.py#L1-L168) [main\.py L1-L206](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/main.py#L1-L206)

## RGCN Approach

 The RGCN approach integrates multiple biological networks using relational graph convolutional networks to predict drug\-drug synergy\. It constructs a heterogeneous graph incorporating drug\-drug interactions, drug\-target interactions, and protein\-protein interactions\.

### RGCN Model Architecture

  The core model is implemented in the `DDs` class which stacks four `RGCNConv` layers with decreasing dimensions\. Each `RGCNConv` layer performs relation\-specific message passing using learnable weight matrices\.

 **Sources:** [R\_model\.py L174-L209](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/R_model.py#L174-L209) [R\_model\.py L73-L147](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/R_model.py#L73-L147)

### Training Process

 The RGCN training process involves subgraph sampling and negative sampling to handle the large\-scale heterogeneous graph:

| Component | Function | Description |
| --- | --- | --- |
| subgraph\_gen | Graph Sampling | Samples subgraphs using GraphSAGE methodology |
| negative\_Sampling | Data Balancing | Generates negative samples with 1:2 ratio |
| train | Model Training | Iterates over multiple subgraphs with cross\-entropy loss |

 The training script uses the `het_gnn_featurizer.DDiFeaturizer` to process multiple data sources into a unified heterogeneous graph representation\.

 **Sources:** [train\.py L44-L92](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/train.py#L44-L92) [graphsage\_sampling\.py L57-L89](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/graphsage_sampling.py#L57-L89) [R\_model\.py L213-L225](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/R_model.py#L213-L225)

## DTSyn Approach

 DTSyn employs a dual\-transformer architecture to capture biological interactions at different granularity levels, combining coarse\-grained and fine\-grained attention mechanisms\.

### DTSyn Model Architecture

  The `TSNet` class implements this dual\-branch architecture where the coarse branch processes aggregated drug and cell representations, while the fine branch incorporates detailed molecular and gene\-level features\.

 **Sources:** [tsnet\.py L25-L118](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/tsnet.py#L25-L118)

### Data Processing Pipeline

 DTSyn uses specialized data processing components for molecular featurization and dataset management:

  The molecular featurization process converts SMILES strings into graph representations with atomic features including element type, degree, hydrogen count, valence, and aromaticity\.

 **Sources:** [utils\_no\_de\.py L49-L77](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/utils_no_de.py#L49-L77) [utils\_no\_de\.py L104-L139](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/utils_no_de.py#L104-L139) [utils\_no\_de\.py L141-L179](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/utils_no_de.py#L141-L179)

## Data Requirements and Formats

 Both approaches require specific datasets with defined formats:

### RGCN Data Requirements

| Dataset | Description | Format | Usage |
| --- | --- | --- | --- |
| DDI | Drug\-drug interactions | CSV with synergy scores | Target labels |
| DTI | Drug\-target interactions | TSV with drug\-protein links | Graph edges |
| PPI | Protein\-protein interactions | TXT with protein links | Graph edges |
| Drug Features | Molecular descriptors | FET format, 2325 dimensions | Node features |

### DTSyn Data Requirements

| Dataset | Description | Format | Usage |
| --- | --- | --- | --- |
| DDI | Drug combinations | CSV with SMILES and labels | Input pairs |
| RNA | Cell line expression | CSV with gene expression | Cell context |
| LINCS | Gene embeddings | CSV with 978\-dimensional vectors | Fine\-grained features |

 **Sources:** [README\.md?plain=1 L20-L40](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/README.md?plain=1#L20-L40) [README\.md?plain=1 L18-L22](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/README.md?plain=1#L18-L22)

## Training and Evaluation

### RGCN Training Command

### DTSyn Training Command

  Both models support comprehensive evaluation metrics including AUC, PRAUC, accuracy, balanced accuracy, precision, recall, and Cohen's kappa score for assessing synergy prediction performance\.

 **Sources:** [README\.md?plain=1 L42-L56](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/RGCN/README.md?plain=1#L42-L56) [README\.md?plain=1 L24-L33](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/README.md?plain=1#L24-L33) [main\.py L81-L92](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_drug_synergy/DTSyn/main.py#L81-L92)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.3-drug-drug-synergy](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.3-drug-drug-synergy) on DeepWiki*