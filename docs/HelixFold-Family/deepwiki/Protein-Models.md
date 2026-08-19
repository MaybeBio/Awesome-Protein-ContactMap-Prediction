# Protein Models

> **Relevant source files**
> * [apps/pretrained_compound/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/README.md?plain=1)
> * [apps/pretrained_compound/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_compound/README_cn.md?plain=1)
> * [apps/pretrained_protein/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/README.md?plain=1)
> * [apps/pretrained_protein/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/README_cn.md?plain=1)
> * [apps/pretrained_protein/tape/README.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1)
> * [apps/pretrained_protein/tape/README_cn.md](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README_cn.md?plain=1)
> * [apps/pretrained_protein/tape/data_gen.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/data_gen.py)
> * [apps/pretrained_protein/tape/eval.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/eval.py)
> * [apps/pretrained_protein/tape/metrics.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/metrics.py)
> * [apps/pretrained_protein/tape/predict.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/predict.py)
> * [apps/pretrained_protein/tape/train.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py)
> * [tutorials/protein_pretrain_and_property_prediction_tutorial.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/protein_pretrain_and_property_prediction_tutorial.ipynb)
> * [tutorials/protein_pretrain_and_property_prediction_tutorial_cn.ipynb](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/tutorials/protein_pretrain_and_property_prediction_tutorial_cn.ipynb)

This document covers the pretrained protein models available in PaddleHelix, focusing on the TAPE (Tasks Assessing Protein Embeddings) framework implementation. These models enable sequence-based protein representation learning through self-supervised pretraining and fine-tuning on downstream tasks. For information about compound representation models, see [Compound Models](/PaddlePaddle/PaddleHelix/4.2-compound-models).

## Overview

The protein models in PaddleHelix implement the TAPE framework, providing pretrained sequence models that can extract meaningful biological representations from protein sequences. The system supports multiple model architectures and enables both pretraining on large unlabeled datasets and fine-tuning on specific downstream tasks.

```mermaid
flowchart TD

PT["ProteinTokenizer"]
PE["ProteinEncoderModel"]
PM["ProteinModel"]
PC["ProteinCriterion"]
TRANS["Transformer"]
LSTM["LSTM"]
RESNET["ResNet"]
TRAIN["train.py"]
EVAL["eval.py"]
PRED["predict.py"]
DG["data_gen.py"]
MET["metrics.py"]

TRANS --> PE
LSTM --> PE
RESNET --> PE
TRAIN --> PM
EVAL --> PM
PRED --> PM
DG --> PT
MET --> PC

subgraph subGraph2 ["Training Pipeline"]
    TRAIN
    EVAL
    PRED
    DG
    MET
end

subgraph subGraph1 ["Model Types"]
    TRANS
    LSTM
    RESNET
end

subgraph subGraph0 ["TAPE Framework Architecture"]
    PT
    PE
    PM
    PC
    PT --> PE
    PE --> PM
    PM --> PC
end
```

Sources: [apps/pretrained_protein/tape/README.md L33-L34](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L33-L34)

 [apps/pretrained_protein/tape/train.py L16](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py#L16-L16)

 [pahelix/model_zoo/protein_sequence_model.py](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/pahelix/model_zoo/protein_sequence_model.py)

## Model Architecture Components

The protein model system consists of several key components that work together to process protein sequences and generate representations.

### Core Model Classes

The `ProteinEncoderModel` serves as the base encoder that generates contextual representations from protein sequences. The `ProteinModel` wraps the encoder and adds task-specific heads for different downstream applications.

```mermaid
flowchart TD

SEQ["Protein Sequence"]
TOK["ProteinTokenizer"]
IDS["Token IDs"]
POS["Position IDs"]
ENC["ProteinEncoderModel"]
HEAD["Task Head"]
OUT["Predictions"]
CRIT["ProteinCriterion"]
LOSS["Loss"]
OPT["AdamW Optimizer"]

IDS --> ENC
POS --> ENC
OUT --> CRIT

subgraph subGraph2 ["Training Components"]
    CRIT
    LOSS
    OPT
    CRIT --> LOSS
    LOSS --> OPT
end

subgraph subGraph1 ["Model Execution"]
    ENC
    HEAD
    OUT
    ENC --> HEAD
    HEAD --> OUT
end

subgraph subGraph0 ["Input Processing"]
    SEQ
    TOK
    IDS
    POS
    SEQ --> TOK
    TOK --> IDS
    TOK --> POS
end
```

Sources: [apps/pretrained_protein/tape/train.py L76-L77](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py#L76-L77)

 [apps/pretrained_protein/tape/predict.py L52-L53](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/predict.py#L52-L53)

### Supported Model Types

The framework supports three different model architectures, each configured through the `model_config` parameter:

| Model Type | Key Parameters | Use Case |
| --- | --- | --- |
| Transformer | `hidden_size`, `layer_num`, `head_num` | Bidirectional context modeling |
| LSTM | `hidden_size`, `layer_num` | Sequential processing |
| ResNet | `hidden_size`, `layer_num`, `filter_num` | Convolutional feature extraction |

Sources: [apps/pretrained_protein/tape/README.md L100-L135](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L100-L135)

## Training and Evaluation Workflow

### Training Process

The training pipeline supports both single-GPU and distributed multi-GPU training. The main training loop is implemented in `train.py` with automatic model checkpointing and evaluation.

```mermaid
flowchart TD

CONFIG["model_config.json"]
TRAIN_DATA["Training Data"]
VALID_DATA["Validation Data"]
LOADER["create_dataloader()"]
MODEL["ProteinEncoderModel + ProteinModel"]
OPT["AdamW Optimizer"]
FORWARD["Forward Pass"]
LOSS_CALC["ProteinCriterion.cal_loss()"]
BACKWARD["loss.backward()"]
UPDATE["optimizer.minimize()"]
EVAL_FUNC["eval() function"]
METRICS["get_metric()"]
SAVE["Save Best Model"]

CONFIG --> LOADER
TRAIN_DATA --> LOADER
VALID_DATA --> LOADER
LOADER --> MODEL
MODEL --> OPT
OPT --> FORWARD
UPDATE --> EVAL_FUNC

subgraph Evaluation ["Evaluation"]
    EVAL_FUNC
    METRICS
    SAVE
    EVAL_FUNC --> METRICS
    METRICS --> SAVE
end

subgraph subGraph0 ["Training Loop"]
    FORWARD
    LOSS_CALC
    BACKWARD
    UPDATE
    FORWARD --> LOSS_CALC
    LOSS_CALC --> BACKWARD
    BACKWARD --> UPDATE
    UPDATE --> FORWARD
end
```

Sources: [apps/pretrained_protein/tape/train.py L114-L155](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py#L114-L155)

 [apps/pretrained_protein/tape/train.py L23-L50](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py#L23-L50)

### Data Processing Pipeline

The data processing system uses bucket-based batching to handle variable-length protein sequences efficiently. Different dataset classes handle different task types.

```mermaid
flowchart TD

PFAM["PfamDataset"]
SEQ["SequenceDataset"]
NORM["NormalDataset"]
BUCKET["Bucket Mechanism"]
PAD["pad_to_max_seq_len()"]
MASK["apply_bert_mask()"]
BATCH["Batch Generation"]
PRETRAIN["Pretraining Tasks"]
SEQ_CLASS["Sequence Classification"]
REGRESSION["Regression Tasks"]

PFAM --> PRETRAIN
SEQ --> SEQ_CLASS
NORM --> REGRESSION
PFAM --> BUCKET
SEQ --> BUCKET
NORM --> BUCKET

subgraph subGraph2 ["Task Types"]
    PRETRAIN
    SEQ_CLASS
    REGRESSION
end

subgraph subGraph1 ["Data Processing"]
    BUCKET
    PAD
    MASK
    BATCH
    BUCKET --> PAD
    PAD --> MASK
    MASK --> BATCH
end

subgraph subGraph0 ["Dataset Classes"]
    PFAM
    SEQ
    NORM
end
```

Sources: [apps/pretrained_protein/tape/data_gen.py L32-L174](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/data_gen.py#L32-L174)

 [apps/pretrained_protein/tape/data_gen.py L29-L30](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/data_gen.py#L29-L30)

## Supported Tasks

### Pretraining Tasks

**Pfam**: Large-scale protein family classification using 30 million protein sequences. Uses masked language modeling for self-supervised learning.

Configuration:

```json
{    "task": "pretrain",    "model_type": "transformer"}
```

Sources: [apps/pretrained_protein/tape/README.md L144-L151](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L144-L151)

### Downstream Tasks

| Task | Type | Classes | Configuration |
| --- | --- | --- | --- |
| Secondary Structure | Sequence Classification | 3 or 8 | `"task": "seq_classification"` |
| Remote Homology | Classification | 1195 | `"task": "classification"` |
| Fluorescence | Regression | N/A | `"task": "regression"` |
| Stability | Regression | N/A | `"task": "regression"` |

Sources: [apps/pretrained_protein/tape/README.md L154-L242](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L154-L242)

### Evaluation Metrics

The system provides task-specific metrics through the `get_metric()` function:

* **PretrainMetric**: Accuracy and perplexity for pretraining tasks
* **ClassificationMetric**: Accuracy for classification tasks
* **RegressionMetric**: MSE and Spearman correlation for regression tasks

Sources: [apps/pretrained_protein/tape/metrics.py L139-L149](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/metrics.py#L139-L149)

## Usage Examples

### Training a Model

```
python train.py \    --train_data ./train_data \    --valid_data ./valid_data \    --model_config ./transformer_config.json \    --use_cuda \    --lr 0.0001
```

### Model Inference

```
cat protein_sequences.txt | python predict.py \    --model ./trained_model.pdparams \    --model_config ./config.json \    --use_cuda
```

Sources: [apps/pretrained_protein/tape/README.md L47-L87](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/README.md?plain=1#L47-L87)

## Integration with PaddleHelix

The protein models integrate with the broader PaddleHelix ecosystem through:

* **Model Zoo**: `pahelix.model_zoo.protein_sequence_model` provides the core model classes
* **Utilities**: `pahelix.utils.protein_tools.ProteinTokenizer` handles sequence tokenization
* **Language Model Tools**: `pahelix.utils.language_model_tools.apply_bert_mask` for pretraining

The models can be loaded and used in downstream applications or fine-tuned for specific tasks within the PaddleHelix framework.

Sources: [apps/pretrained_protein/tape/train.py L16-L17](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/train.py#L16-L17)

 [apps/pretrained_protein/tape/data_gen.py L25-L26](https://github.com/PaddlePaddle/PaddleHelix/blob/7fdd7a18/apps/pretrained_protein/tape/data_gen.py#L25-L26)