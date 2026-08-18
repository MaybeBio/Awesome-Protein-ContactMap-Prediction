---
title: "Training System"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/5-training-system
---
# Training System

## Data Preparation for Training

 Before running training, you need to prepare the necessary data\. The Boltz training system requires pre\-processed structural data and MSA files\.

### Pre\-processed Data Sources

 The training system uses several pre\-processed data sources:

 1. **RCSB \(PDB\) Structures**: Processed protein structures from the Protein Data Bank
2. **RCSB MSA Files**: Multiple sequence alignments for PDB structures
3. **OpenFold Structures**: Optional distillation dataset
4. **OpenFold MSA Files**: MSAs for the OpenFold dataset
5. **Ligand Symmetry Information**: Symmetry data for small molecules

### Data Processing Pipeline

  Sources:

 - [training\.md?plain=1 L113-L263](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L113-L263)
- [requirements\.txt L1-L5](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/requirements.txt#L1-L5)

### External Dependencies

 The data processing pipeline requires several external dependencies:

 1. **MMSeqs2**: For sequence clustering
2. **Redis**: For sharing large dictionaries across workers
3. **Additional Python packages**: Listed in `scripts/process/requirements.txt`

 Sources:

 - [training\.md?plain=1 L132-L136](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L132-L136)
- [requirements\.txt L1-L5](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/process/requirements.txt#L1-L5)

# Training System

> **Relevant source files**
> - [scripts/train/configs/confidence\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml)
> - [scripts/train/configs/full\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml)
> - [scripts/train/configs/structure\.yaml](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

 The Training System in Boltz provides a framework for training the Boltz1 and Boltz2 models for biomolecular structure prediction\. This system handles the complete training workflow, from data preparation and loading to model configuration, training loop execution, and checkpoint management\. The Training System leverages PyTorch Lightning for efficient multi\-GPU training and supports different training configurations, including structure\-only training, full model training, and confidence\-only training\.

 For information about data preparation specifically, see [Data Preparation](https://deepwiki.com/jwohlwend/boltz/5.1-training-configuration)\. For information about running predictions with a trained model, see [Prediction Pipeline](https://deepwiki.com/jwohlwend/boltz/2-prediction-pipeline)\.

## Training System Architecture

 The Training System consists of several interconnected components that work together to train the Boltz models\. The following diagram illustrates the architecture of the Training System:

  Sources:

 - [train\.py L80-L231](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L80-L231)
- [training\.md?plain=1 L40-L109](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L40-L109)

## Data Flow

 The following diagram illustrates the data flow during the training process:

  Sources:

 - [train\.py L125-L130](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L125-L130)
- [training\.md?plain=1 L96-L109](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L96-L109)
- [structure\.yaml L23-L35](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L23-L35)

## Configuration System

 The Training System uses YAML configuration files to specify all aspects of training, including data sources, model architecture, and training parameters\. The configuration system is designed to be flexible and allows for easy customization\.

### Configuration Hierarchy

  Sources:

 - [train\.py L24-L77](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L24-L77)
- [train\.py L91-L100](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L91-L100)

### Key Configuration Options

 The training configuration files contain several key sections:

| Section | Description | Key Parameters |
| --- | --- | --- |
| trainer | PyTorch Lightning Trainer configuration | accelerator, devices, precision, gradient\_clip\_val, accumulate\_grad\_batches |
| data | Data loading and processing configuration | datasets, filters, tokenizer, featurizer, max\_tokens, max\_atoms, max\_seqs |
| model | Model architecture and parameters | atom\_s, token\_s, embedder\_args, msa\_args, pairformer\_args, score\_model\_args |
| training\_args | Training hyperparameters | recycling\_steps, sampling\_steps, diffusion\_multiplicity, learning rate parameters |
| diffusion\_process\_args | Diffusion model parameters | sigma\_min, sigma\_max, sigma\_data, rho, noise\_scale |
| steering\_args | Physical guidance parameters | fk\_steering, num\_particles, fk\_lambda, fk\_resampling\_interval |
| wandb | Weights & Biases logging configuration | name, project, entity |

 Sources:

 - [full\.yaml L1-L200](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L1-L200)
- [confidence\.yaml L1-L201](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L1-L201)
- [structure\.yaml L1-L195](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L1-L195)

## Training Modes

 The Training System supports three different training modes, each with specialized configuration files:

### Structure\-Only Training

 Structure\-only training focuses exclusively on the structure prediction module without training confidence prediction\. This is typically used for the initial training phase to develop the core structure prediction capabilities\.

 Key configuration:

  Sources:

 - [structure\.yaml L127](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L127-L127)

### Full Model Training

 Full model training trains all components of the Boltz model, including both structure prediction and confidence prediction modules simultaneously\. This is used when you want to train a complete model from scratch or fine\-tune both components together\.

 Key configuration:

  Sources:

 - [full\.yaml L128-L129](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L128-L129)

### Confidence\-Only Training

 Confidence\-only training focuses exclusively on training the confidence prediction module, typically starting from a pre\-trained structure model\. This allows for specialization of the confidence metrics without disturbing the structure prediction capabilities\.

 Key configuration:

  Sources:

 - [confidence\.yaml L17](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L17-L17)
- [confidence\.yaml L129-L130](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L129-L130)

## Data Module

 The data module is responsible for loading, processing, and batching data for training\. It uses PyTorch Lightning's DataModule interface to integrate with the training loop\.

### Data Module Components

  Sources:

 - [train\.py L125-L127](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L125-L127)
- [full\.yaml L23-L76](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L23-L76)
- [structure\.yaml L36-L44](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L36-L44)

### Dataset Configuration

 The data module can be configured to use multiple datasets, each with specific sampling probabilities, filters, and processing options:

  You can also configure multiple datasets with different sampling probabilities:

  Sources:

 - [full\.yaml L24-L35](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L24-L35)
- [training\.md?plain=1 L68-L96](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L68-L96)

## Running Training

### Prerequisites

 Before running training, you need to:

 1. Download the pre\-processed data \(~250GB total storage required\):  - RCSB \(PDB\) structures: `rcsb_processed_targets.tar` - RCSB MSA files: `rcsb_processed_msa.tar` - OpenFold structures \(optional\): `openfold_processed_targets.tar` - OpenFold MSA files \(optional\): `openfold_processed_msa.tar` - Ligand symmetry information: `symmetry.pkl`
2. Modify the configuration file to set:  - `output`: Path to output directory - `data.datasets[].target_dir`: Path to processed structure files - `data.datasets[].msa_dir`: Path to processed MSA files - `data.symmetries`: Path to symmetry file - `resume`: Path to checkpoint file \(if resuming\) - `pretrained`: Path to pretrained model \(if applicable\) - `trainer.devices`: Number or list of GPU devices to use - `trainer.accumulate_grad_batches`: Gradient accumulation steps

 Sources:

 - [training\.md?plain=1 L5-L40](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L5-L40)
- [training\.md?plain=1 L42-L66](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L42-L66)
- [structure\.yaml L1-L7](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L1-L7)

### Training Command

 The training script is invoked using the following command pattern:

```
python scripts/train/train.py <config_file> [command_line_overrides]
```

 For example:

```
# For debugging/testing
python scripts/train/train.py scripts/train/configs/structure.yaml debug=1

# For structure-only model training
python scripts/train/train.py scripts/train/configs/structure.yaml

# For full model training (structure + confidence)
python scripts/train/train.py scripts/train/configs/full.yaml

# For confidence-only model training
python scripts/train/train.py scripts/train/configs/confidence.yaml
```

 Sources:

 - [training\.md?plain=1 L100-L110](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/training.md?plain=1#L100-L110)
- [train\.py L238-L241](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L238-L241)

### Training Process Flow

  Sources:

 - [train\.py L206-L231](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L206-L231)
- [train\.py L131-L166](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L131-L166)
- [confidence\.yaml L22](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L22-L22)

## Implementation Details

### Training Script

 The main training script `train.py` is responsible for:

 1. Loading and merging configuration from YAML file and command line overrides
2. Instantiating data and model modules
3. Setting up PyTorch Lightning trainer with appropriate strategy \(DDP for multi\-GPU\)
4. Configuring checkpointing and logging
5. Running the training or validation process

 The script also includes special handling for pre\-trained models, including the ability to load confidence module weights from the main model "trunk" when `load_confidence_from_trunk` is set to `true`\.

 Sources:

 - [train\.py L80-L241](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L80-L241)
- [train\.py L131-L166](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L131-L166)

### Model Training Configuration

 The model training configuration includes settings for:

 1. Structure prediction module training
2. Confidence prediction module training
3. Diffusion process parameters
4. Learning rate scheduling
5. Loss function weights
6. Recycling steps \(iterative refinement\)

 Key training parameters include:

| Parameter | Description | Typical Value |
| --- | --- | --- |
| recycling\_steps | Number of iterative refinement steps | 3 |
| sampling\_steps | Number of diffusion sampling steps | 20\-200 |
| diffusion\_multiplicity | Number of parallel diffusion processes | 16 |
| diffusion\_samples | Number of diffusion samples per batch | 1\-5 |
| confidence\_loss\_weight | Weight for confidence prediction loss | 1e\-4 to 3e\-3 |
| diffusion\_loss\_weight | Weight for diffusion loss | 4\.0 |
| distogram\_loss\_weight | Weight for distogram prediction loss | 3e\-2 |
| lr\_scheduler | Learning rate scheduler type | "af3" |
| max\_lr | Maximum learning rate | 0\.0018 |

 Sources:

 - [full\.yaml L145-L164](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L145-L164)
- [confidence\.yaml L146-L165](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/confidence.yaml#L146-L165)
- [structure\.yaml L142-L158](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L142-L158)

## Advanced Usage

### Multi\-GPU Training

 The Training System automatically configures distributed data parallel \(DDP\) training when multiple GPUs are specified:

  The `DDPStrategy` with configurable `find_unused_parameters` is automatically used for multi\-GPU setups\.

 Sources:

 - [train\.py L205-L209](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L205-L209)
- [structure\.yaml L1-L7](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L1-L7)

### Resuming Training

 Training can be resumed from a checkpoint by specifying the `resume` parameter:

  The trainer will automatically restore the model state, optimizer state, and training progress from the checkpoint\.

 Sources:

 - [train\.py L231-L235](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/train.py#L231-L235)
- [structure\.yaml L17](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/structure.yaml#L17-L17)

### Performance Optimization

 Several configuration options can be adjusted to optimize training performance:

 1. Batch size and gradient accumulation
2. Precision \(fp32, bf16, fp16\)
3. Activation checkpointing and CPU offloading
4. Number of workers for data loading

 Sources:

 - [full\.yaml L1-L7](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L1-L7)
- [full\.yaml L60-L61](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L60-L61)
- [full\.yaml L104-L106](https://github.com/jwohlwend/boltz/blob/cb04aecc/scripts/train/configs/full.yaml#L104-L106)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/5-training-system](https://deepwiki.com/jwohlwend/boltz/5-training-system) on DeepWiki*