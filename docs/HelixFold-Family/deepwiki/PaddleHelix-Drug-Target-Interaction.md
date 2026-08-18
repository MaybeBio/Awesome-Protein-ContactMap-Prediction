---
title: "Drug-Target Interaction"
source: deepwiki.com
owner: PaddlePaddle
repo: PaddleHelix
url: https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction
---
# Drug\-Target Interaction

# Drug\-Target Interaction

> **Relevant source files**
> - [apps/drug\_target\_interaction/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/README.md?plain=1)
> - [apps/drug\_target\_interaction/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/README_cn.md?plain=1)
> - [apps/drug\_target\_interaction/batchdta/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/README.md?plain=1)
> - [apps/drug\_target\_interaction/batchdta/pairwise/DeepDTA/model\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/model.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/DeepDTA/run\_pairwise\_deepdta\_CV\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/run_pairwise_deepdta_CV.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/DeepDTA/run\_pairwise\_deepdta\_bindingDB\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/run_pairwise_deepdta_bindingDB.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/DeepDTA/utils\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/utils.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/GraphDTA/models/gat\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/GraphDTA/models/gat.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/GraphDTA/models/gat\_gcn\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/GraphDTA/models/gat_gcn.py)
> - [apps/drug\_target\_interaction/batchdta/pairwise/GraphDTA/models/gcn\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/GraphDTA/models/gcn.py)
> - [apps/drug\_target\_interaction/graph\_dta/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1)
> - [apps/drug\_target\_interaction/graph\_dta/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README_cn.md?plain=1)
> - [apps/drug\_target\_interaction/graph\_dta/scripts/preprocess\_data\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py)
> - [apps/drug\_target\_interaction/graph\_dta/scripts/train\.sh](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/train.sh)
> - [apps/drug\_target\_interaction/graph\_dta/train\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/train.py)
> - [apps/drug\_target\_interaction/moltrans\_dti/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1)
> - [apps/pretrained\_compound/info\_graph/README\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README.md?plain=1)
> - [apps/pretrained\_compound/info\_graph/README\_cn\.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/README_cn.md?plain=1)
> - [apps/pretrained\_compound/info\_graph/classifier\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/classifier.py)
> - [apps/pretrained\_compound/info\_graph/scripts/preprocess\_data\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/scripts/preprocess_data.py)
> - [apps/pretrained\_compound/info\_graph/scripts/unsupervised\_pretrain\.sh](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/scripts/unsupervised_pretrain.sh)
> - [apps/pretrained\_compound/info\_graph/src/model\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/src/model.py)
> - [apps/pretrained\_compound/info\_graph/unsupervised\_pretrain\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/unsupervised_pretrain.py)

 This document covers the drug\-target interaction \(DTI\) prediction capabilities in PaddleHelix, including multiple neural network architectures, training frameworks, and evaluation approaches for predicting binding affinity between drug compounds and target proteins\.

 For compound representation learning without target interaction, see [Compound Representation Learning](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.1-compound-representation-learning)\. For drug synergy prediction involving multiple drugs, see [Drug\-Drug Synergy](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.3-drug-drug-synergy)\.

## Overview

 Drug\-target interaction prediction is a fundamental task in computational drug discovery that estimates the binding affinity between small molecule compounds and target proteins\. PaddleHelix provides several state\-of\-the\-art approaches including graph neural networks, convolutional architectures, and transformer models, along with specialized training frameworks to handle real\-world data challenges\.

## System Architecture

### Model Architecture Overview

  Sources: [README\.md?plain=1 L1-L12](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/README.md?plain=1#L1-L12) [preprocess\_data\.py L32-L34](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py#L32-L34) [train\.py L27-L29](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/train.py#L27-L29)

### Core Components and Code Mapping

  Sources: [apps/drug\_target\_interaction/graph\_dta/src/model\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/src/model.py) [apps/drug\_target\_interaction/graph\_dta/src/data\_gen\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/src/data_gen.py) [gcn\.py L8-L32](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/GraphDTA/models/gcn.py#L8-L32) [README\.md?plain=1 L134-L149](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1#L134-L149)

## Model Architectures

### GraphDTA Models

 GraphDTA represents drugs as molecular graphs and uses graph neural networks for feature extraction\. The system supports multiple GNN architectures:

| Architecture | Configuration File | Description |
| --- | --- | --- |
| GCN | fix\_prot\_len\_gcn\_config\.json | Graph Convolutional Network |
| GAT | fix\_prot\_len\_gat\_config\.json | Graph Attention Network |
| GAT\-GCN | fix\_prot\_len\_gat\_gcn\_config\.json | Hybrid attention \+ convolution |
| GIN | fix\_prot\_len\_gin\_config\.json | Graph Isomorphism Network |

 The `DTAModel` class in [apps/drug\_target\_interaction/graph\_dta/src/model\.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/src/model.py) implements the core architecture, taking molecular graphs and protein sequences as input through the `DTACollateFunc` collation function\.

 **Key Classes:**

 - `DTADataset`: Handles data loading and preprocessing
- `DTACollateFunc`: Collates molecular graphs and protein sequences
- `DTAModelCriterion`: Loss function for training

 Sources: [README\.md?plain=1 L81-L86](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1#L81-L86) [train\.py L27-L29](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/train.py#L27-L29)

### DeepDTA Models

 DeepDTA uses 1D convolutional neural networks for both drug and protein sequence processing\. The architecture consists of:

 1. **Drug Branch**: CNN over SMILES character sequences
2. **Protein Branch**: CNN over protein amino acid sequences
3. **Fusion Layer**: Concatenated features fed through dense layers

 The implementation supports both pointwise training \(individual affinity prediction\) and pairwise training \(relative ranking\) through the BatchDTA framework\.

 Sources: [model\.py L20-L110](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/model.py#L20-L110)

### MolTrans Models

 MolTrans leverages transformer architecture for drug\-target interaction prediction with dual\-tower design:

  The system supports both classification and regression tasks through separate training scripts \(`train_cls.py` and `train_reg.py`\)\.

 Sources: [README\.md?plain=1 L136-L149](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1#L136-L149)

## BatchDTA Training Framework

 BatchDTA addresses batch effects in drug\-target affinity datasets by using implicit batch alignment during training\. The framework supports multiple backbone models and training strategies:

### Pairwise Training Approach

  The pairwise approach trains models to predict relative rankings between drug\-target pairs sharing the same target protein, helping to mitigate batch effects across different assay conditions\.

 **Key Functions:**

 - `group_by()`: Groups data by target protein ID
- `get_pairs()`: Generates ordered pairs with score differences \> threshold
- `sample_pairs()`: Samples training pairs with filtering
- `Data_Encoder_flow`: DataLoader for pairwise training data

 Sources: [README\.md?plain=1 L5-L9](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/README.md?plain=1#L5-L9) [utils\.py L30-L99](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/utils.py#L30-L99)

## Datasets and Data Processing

### Supported Datasets

| Dataset | Drugs | Targets | Metric | Description |
| --- | --- | --- | --- | --- |
| Davis | 72 | 442 | Kd \(nM\) | Kinase inhibitor selectivity |
| KIBA | 2,116 | 229 | KIBA score | Integrated Ki, Kd, IC50 |
| BindingDB | ~11,000 | 110 | Multiple | Protein\-ligand binding |
| ChEMBL | \>1\.6M | ~11,000 | Various | Bioactivity database |
| BioSNAP | Variable | Variable | Binary | Biomedical networks |

### Data Preprocessing Pipeline

 The preprocessing pipeline converts raw datasets into training\-ready format:

 1. **Molecular Processing**: SMILES → RDKit Mol → Graph features via `mol_to_graph_data()`
2. **Protein Processing**: Amino acid sequences → Token IDs via `ProteinTokenizer`
3. **Data Splitting**: Train/validation/test splits with protein\-based or random partitioning
4. **Format Conversion**: Save processed data as NPZ files via `save_data_list_to_npz()`

 **Key Processing Functions:**

 - [preprocess\_data\.py L92-L94](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py#L92-L94): Molecular graph conversion
- [preprocess\_data\.py L97-L99](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py#L97-L99): Protein tokenization
- [preprocess\_data\.py L110-L111](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py#L110-L111): Data serialization

 Sources: [README\.md?plain=1 L26-L36](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1#L26-L36) [README\.md?plain=1 L94-L111](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1#L94-L111)

## Training and Evaluation

### Training Scripts and Usage

 Each model provides specific training scripts with standardized interfaces:

 **GraphDTA Training:**

  **MolTrans Training:**

  **BatchDTA Training:**

  Sources: [README\.md?plain=1 L100-L106](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1#L100-L106) [README\.md?plain=1 L155-L175](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1#L155-L175) [README\.md?plain=1 L49-L55](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/README.md?plain=1#L49-L55)

### Evaluation Metrics

 The system uses multiple evaluation metrics appropriate for different task types:

 **Regression Tasks:**

 - **MSE \(Mean Squared Error\)**: Lower is better
- **CI \(Concordance Index\)**: Higher is better, measures ranking quality

 **Classification Tasks:**

 - **AUROC \(Area Under ROC Curve\)**: Higher is better

 **Implementation:**

 - CI calculation: `concordance_index()` from lifelines package
- MSE calculation: Standard numpy operations
- Evaluation function: `model_eval()` in [utils\.py L300-L351](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/pairwise/DeepDTA/utils.py#L300-L351)

### Performance Results

 **Davis Dataset Performance:**

| Method | MSE | CI | AUROC |
| --- | --- | --- | --- |
| GraphDTA \(GIN\) | 0\.242 | 0\.889 | \- |
| MolTrans | 0\.199 | 0\.901 | 0\.912 |
| BatchDTA \(avg improvement\) | \- | \+4\.0% | \- |

 **KIBA Dataset Performance:**

| Method | MSE | CI |
| --- | --- | --- |
| GraphDTA \(GAT\-GCN\) | 0\.142 | 0\.895 |
| MolTrans | 0\.132 | 0\.898 |

 Sources: [README\.md?plain=1 L111-L128](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/README.md?plain=1#L111-L128) [README\.md?plain=1 L180-L198](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/moltrans_dti/README.md?plain=1#L180-L198) [README\.md?plain=1 L9](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/batchdta/README.md?plain=1#L9-L9)

## Integration with PaddleHelix

 The DTI system integrates with the broader PaddleHelix ecosystem through:

 1. **Compound Featurization**: Uses `pahelix.utils.compound_tools.mol_to_graph_data` for molecular graph conversion
2. **Protein Processing**: Leverages `pahelix.utils.protein_tools.ProteinTokenizer` for sequence encoding
3. **Data Management**: Built on `pahelix.datasets.inmemory_dataset.InMemoryDataset` infrastructure
4. **Graph Learning**: Utilizes PGL \(PaddlePaddle Graph Learning\) for GNN implementations

 The modular design allows easy extension and integration with other PaddleHelix components for end\-to\-end drug discovery pipelines\.

 Sources: [preprocess\_data\.py L32-L34](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/drug_target_interaction/graph_dta/scripts/preprocess_data.py#L32-L34) [unsupervised\_pretrain\.py L39](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/info_graph/unsupervised_pretrain.py#L39-L39)

---
*Source: [https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction](https://deepwiki.com/PaddlePaddle/PaddleHelix/3.2.2-drug-target-interaction) on DeepWiki*