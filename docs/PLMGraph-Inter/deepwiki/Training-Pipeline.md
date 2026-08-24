# Training Pipeline

> **Relevant source files**
> * [model.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/model.py)
> * [train.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py)

The Training Pipeline in PLMGraph-Inter is responsible for training the ResNet-GVP neural network model that predicts protein-protein interactions (PPIs). This page details the complete training process, including data preparation, the loss function, optimization methods, evaluation metrics, and model saving strategy. For details about the model architecture itself, see [Model Architecture](/ChengfeiYan/PLMGraph-Inter/5-model-architecture).

## Training Pipeline Overview

The training pipeline integrates graph-based protein structure representations with a specialized residue contact prediction framework. It processes paired protein data, extracts geometric and sequential features, and trains a model to predict inter-protein residue contacts.


Sources: [train.py L149-L247](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L149-L247)

## Training Data Preparation

The training pipeline uses datasets of protein pairs, including both homodimers and heterodimers. The data is split into training and validation sets, with approximately 1050 samples for validation and the rest for training.

```mermaid
flowchart TD

homo["Homodimers Dataset"]
hetero["Heterodimers Dataset"]
combined["Combined Training Dataset"]
shuffle["Random Shuffling"]
split["Train/Validation Split"]
train_list["Training List"]
valid_list["Validation List"]
trainset["Dataset(train_all_path, train_list)"]
validset["Dataset(train_all_path, valid_list)"]
train_loader["DataLoader(<br>batch_size=1,<br>shuffle=True,<br>num_workers=6,<br>prefetch_factor=3,<br>persistent_workers=True)"]
valid_loader["DataLoader(<br>batch_size=1,<br>shuffle=True,<br>num_workers=6,<br>prefetch_factor=3,<br>persistent_workers=True)"]

combined --> shuffle
train_list --> trainset
valid_list --> validset

subgraph subGraph2 ["DataLoader Configuration"]
    trainset
    validset
    train_loader
    valid_loader
    trainset --> train_loader
    validset --> valid_loader
end

subgraph subGraph1 ["Data Splitting"]
    shuffle
    split
    train_list
    valid_list
    shuffle --> split
    split --> train_list
    split --> valid_list
end

subgraph subGraph0 ["Data Sources"]
    homo
    hetero
    combined
    homo --> combined
    hetero --> combined
end
```

Sources: [train.py L83-L107](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L83-L107)

### Dataset Structure

The dataset contains paired protein feature data including:

* Protein graph data (nodes, edges, edge indices)
* Paired 2D features (p2d)
* Contact maps and masking maps

Sources: [train.py L165-L180](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L165-L180)

## Loss Function

The training pipeline uses a specialized loss function called `ppi_loss` designed specifically for protein-protein interaction prediction. This loss function combines aspects of binary cross-entropy with customizable weighting.

```mermaid
classDiagram
    class ppi_loss {
        +init(alpha, inter, clamp, reduction)
        +forward(Input, Label, mask)
    }
    class nn.Module {
        +forward()
    }
    nn.Module <|-- ppi_loss
```

The loss function accepts:

* `Input`: The predicted probability map (values between 0 and 1)
* `Label`: The ground truth contact map
* `mask`: A mask to focus the loss computation on relevant residue pairs

When alpha is specified, the loss introduces weights that emphasize different regions of the prediction space:

* For positive examples (Label=1): `-alpha * (2-Input)² * Label * log(Input)`
* For negative examples (Label=0): `-(1-alpha) * (1+Input)³ * (1-Label) * log(1-Input)`

This weighting scheme encourages the model to be more confident in its predictions.

Sources: [train.py L46-L79](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L46-L79)

## Training Process

The training process runs for a specified number of epochs (default: 100), alternating between training and validation phases in each epoch.

### Model Initialization

The model is initialized as a ResNet18 with GVP components:

```
model = resnet18().to(device)criterion_ppi = ppi_loss(alpha=None, reduction='sum')optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', eps=1e-6, patience=3, factor=0.1)
```

Sources: [train.py L112-L121](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L112-L121)

### Training Loop Structure

```mermaid
flowchart TD

start["Start Training Loop"]
epoch_loop["For each epoch (0-99)"]
phase_loop["For each phase (train, valid)"]
set_mode["Is training phase?"]
train_mode["Set model.train()<br>Use train_loader"]
eval_mode["Set model.eval()<br>Use valid_loader"]
data_loop["For each data batch"]
prep_data["Prepare protein data:<br>- Move tensors to device<br>- Extract nodes, edges, indices<br>- Prepare p2d, mask_map, contact_map"]
subsample["Optional subsampling:<br>If protein > max_aa (400),<br>take random subset"]
forward_grad["Is training phase?"]
train_forward["Forward with gradient tracking<br>Calculate loss<br>Backward pass<br>Optimizer step"]
valid_forward["Forward without gradient tracking<br>Calculate loss"]
calc_metrics["Calculate Top-K statistics"]
log_progress["Batch count % 100 == 0<br>or last batch?"]
print_stats["Print progress statistics"]
cont_loop["Continue loop"]
end_batch["End batch loop"]
phase_end["Is validation phase?"]
scheduler_update["Update learning rate scheduler<br>Record validation statistics"]
record_train["Record training statistics"]
save_models["Save models based on metrics"]
end_epoch["End epoch loop"]
next_epoch["More epochs?"]
end_training["End training"]

start --> epoch_loop
epoch_loop --> phase_loop
phase_loop --> set_mode
set_mode --> train_mode
set_mode --> eval_mode
train_mode --> data_loop
eval_mode --> data_loop
data_loop --> prep_data
prep_data --> subsample
subsample --> forward_grad
forward_grad --> train_forward
forward_grad --> valid_forward
train_forward --> calc_metrics
valid_forward --> calc_metrics
calc_metrics --> log_progress
log_progress --> print_stats
log_progress --> cont_loop
print_stats --> end_batch
cont_loop --> end_batch
end_batch --> phase_end
phase_end --> scheduler_update
phase_end --> record_train
scheduler_update --> save_models
record_train --> save_models
save_models --> end_epoch
end_epoch --> next_epoch
next_epoch --> epoch_loop
next_epoch --> end_training
```

Sources: [train.py L149-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L149-L201)

### Data Processing in Training Loop

For each batch, the following processing occurs:

1. Protein data (nodeA, edgeA, edge_indexA, nodeB, edgeB, edge_indexB) is prepared and moved to the GPU
2. Paired 2D features (p2d), mask map, and contact map are processed
3. For large proteins (>400 residues), a random subset of positions is sampled to manage memory usage
4. Forward pass through the model produces predicted contact probabilities
5. Loss is calculated using the ppi_loss function
6. For training phase only: * Backward pass computes gradients * Optimizer updates model parameters

Sources: [train.py L165-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L165-L201)

## Evaluation Metrics

The model is evaluated using Top-K statistics, which measure the accuracy of contact prediction by looking at the top K predicted contacts.

```mermaid
flowchart TD

input["Inputs:<br>- pred_map: Predicted contact probabilities<br>- contact_map: Ground truth contacts<br>- Topk_list: List of K values"]
processing["Processing:<br>1. Calculate protein length L<br>2. For each K value:<br>- Get top K predictions<br>- Count true positives"]
output["Output:<br>Array of precision values for each K"]
metrics["Metrics:<br>- L/5, L/10, L/20 (L = min protein length)<br>- Fixed values: 50, 20, 10, 5, 1"]

metrics --> input

subgraph subGraph1 ["Top-K Metrics Used"]
    metrics
end

subgraph subGraph0 ["top_statistics_ppi Function"]
    input
    processing
    output
    input --> processing
    processing --> output
end
```

The top_statistics_ppi function calculates:

* Precision at various K values (where K can be fixed numbers or relative to protein length)
* For relative thresholds (e.g., L/5), K is calculated as a fraction of the minimum protein length

Sources: [train.py L20-L43](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L20-L43)

 [train.py L127-L131](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L127-L131)

## Model Saving Strategy

The training pipeline implements a comprehensive model saving strategy that preserves:

1. The best performing model for each Top-K metric
2. The model with the lowest validation loss
3. Checkpoints at each epoch

```mermaid
flowchart TD

eval["Evaluation Metrics Calculated"]
check_topk["For each Top-K metric"]
compare_best["Is current accuracy > highest?"]
save_topk["Save model:<br>final_model/GCN1_5_{metric}.pth"]
skip_topk["Skip saving"]
check_loss["Is current loss < minimum loss?"]
save_loss["Save model:<br>final_model/GCN1_5_minloss.pth"]
skip_loss["Skip saving"]
save_checkpoint["Save epoch checkpoint:<br>final_model/GCN1_5_{epoch}.pkl"]
next_epoch["Proceed to next epoch"]

eval --> check_topk
check_topk --> compare_best
compare_best --> save_topk
compare_best --> skip_topk
eval --> check_loss
check_loss --> save_loss
check_loss --> skip_loss
eval --> save_checkpoint
save_topk --> next_epoch
skip_topk --> next_epoch
save_loss --> next_epoch
skip_loss --> next_epoch
save_checkpoint --> next_epoch
```

The saved models include:

* Models with best performance for each Top-K metric (L/5, L/10, L/20, 50, 20, 10, 5, 1)
* Model with minimum validation loss
* Checkpoint at each epoch

Sources: [train.py L230-L247](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/train.py#L230-L247)

## Code References

The primary code files for the training pipeline are:

* `train.py`: Contains the main training loop, loss function, and evaluation metrics
* `model.py`: Contains the ResNet-GVP model architecture used in training

The training pipeline integrates with the model architecture defined in the ResNet and GVP components, which are documented in the [Model Architecture](/ChengfeiYan/PLMGraph-Inter/5-model-architecture) page.