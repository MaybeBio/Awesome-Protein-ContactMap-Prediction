---
title: "Confidence Prediction"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction
---
# Confidence Prediction

# Confidence Prediction

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

## Purpose and Scope

 The Confidence Prediction system in Boltz estimates the reliability of predicted structures by computing various quality metrics\. This module predicts per\-residue and pairwise confidence scores that indicate how accurate different regions of the predicted structure are likely to be\. These metrics are essential for ranking multiple predictions and identifying well\-modeled versus uncertain regions\.

 For information about the affinity prediction module, see [Affinity Prediction](https://deepwiki.com/jwohlwend/boltz/3.6-affinity-prediction)\. For details on the diffusion\-based structure generation that produces the coordinates, see [Diffusion Process](https://deepwiki.com/jwohlwend/boltz/3.4-diffusion-process)\.

 **Sources:** [prediction\.md?plain=1 L198-L226](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L198-L226) [boltz2\.py L304-L319](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L304-L319)

## Architecture Overview

 The confidence prediction system operates as a post\-processing module that analyzes the model's internal representations and predicted coordinates to estimate confidence\. It runs after structure generation and produces multiple types of confidence scores\.

  **Sources:** [boltz2\.py L585-L606](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L585-L606) [boltz1\.py L379-L398](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L379-L398)

## Confidence Module

### Boltz\-1 vs Boltz\-2 Confidence Modules

 Boltz uses two versions of the confidence module depending on the model:

| Aspect | Boltz\-1 | Boltz\-2 |
| --- | --- | --- |
| Module Path | boltz\.model\.modules\.confidence | boltz\.model\.modules\.confidencev2 |
| Token\-level Confidence | Configurable | Default enabled |
| PAE Computation | Optional \(compute\_pae flag\) | Controlled by alpha\_pae parameter |
| Trunk Imitation | Optional mode | Not used |
| Diffusion Features | Can use s\_diffusion | Not used |

 **Sources:** [boltz1\.py L234-L256](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L234-L256) [boltz2\.py L304-L319](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L304-L319)

### Module Initialization

 The `ConfidenceModule` is initialized with the following key parameters:

  **Sources:** [boltz2\.py L305-L319](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L305-L319)

### Forward Pass

 The confidence module is called during the forward pass after structure generation:

  The confidence module receives detached \(no gradient\) versions of the trunk embeddings and predicted coordinates to prevent gradient flow during training\.

 **Sources:** [boltz2\.py L585-L606](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L585-L606)

## Confidence Metrics

### pLDDT \(Predicted Local Distance Difference Test\)

 **pLDDT** is a per\-token metric that estimates the local quality of the prediction\. It predicts what LDDT score the structure would achieve in that region\.

 - **Range:** \[0, 1\] where higher is better
- **Interpretation:** - > 0\.9: Very high confidence - 0\.7\-0\.9: Confident - 0\.5\-0\.7: Low confidence - < 0\.5: Very low confidence \(should not be trusted\)

 **Aggregated Metrics:**

 - `complex_plddt`: Average pLDDT across all tokens
- `complex_iplddt`: Interface\-weighted average, emphasizing binding regions

 **Sources:** [prediction\.md?plain=1 L198-L226](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L198-L226) [boltz2\.py L1084-L1097](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1084-L1097)

### PAE \(Predicted Aligned Error\)

 **PAE** is a pairwise metric measuring the expected position error between pairs of residues/tokens when the predicted and true structures are aligned on one of them\.

 - **Range:** Angstroms \(lower is better\)
- **Output:** N\_tokens × N\_tokens matrix
- **Use cases:** - Identifying well\-defined relative positions - Domain boundary detection - Multi\-chain interface confidence

 **Derived Metrics:**

 - `ptm`: Predicted TM\-score for the entire complex
- `iptm`: Interface TM\-score \(only interfaces between chains\)
- `ligand_iptm`: Protein\-ligand interface TM\-score
- `protein_iptm`: Protein\-protein interface TM\-score
- `chains_ptm`: Per\-chain TM\-scores
- `pair_chains_iptm`: Pairwise interface TM\-scores between chains

 **Sources:** [prediction\.md?plain=1 L210-L223](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L210-L223) [boltz2\.py L1101-L1106](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1101-L1106)

### PDE \(Predicted Distance Error\)

 **PDE** is a pairwise metric predicting the distance error between pairs of tokens\.

 - **Range:** Angstroms \(lower is better\)
- **Output:** N\_tokens × N\_tokens matrix
- **Interpretation:** Expected absolute error in the distance between two residues

 **Aggregated Metrics:**

 - `complex_pde`: Average PDE across all token pairs
- `complex_ipde`: Interface\-weighted PDE, focusing on binding regions

 The PDE provides complementary information to PAE and can be particularly useful for evaluating local geometry\.

 **Sources:** [prediction\.md?plain=1 L208-L209](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L208-L209) [boltz2\.py L1098-L1099](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1098-L1099)

### PTM and iPTM

 **PTM \(Predicted Template Modeling Score\)** and **iPTM \(Interface PTM\)** are derived from the PAE matrix and estimate overall structural quality\.

  **Usage in Ranking:**

 The final confidence score used to rank predictions is computed as:

```
confidence_score = 0.8 × complex_plddt + 0.2 × (iptm if available else ptm)
```

 **Sources:** [prediction\.md?plain=1 L201-L203](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L201-L203) [boltz2\.py L1085-L1094](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1085-L1094)

## Integration with Model Architecture

### Boltz\-2 Integration

  **Training Configuration:**

 The confidence loss is weighted and combined with structure losses:

| Loss Component | Default Weight |
| --- | --- |
| Diffusion Loss | 4\.0 |
| Distogram Loss | 0\.03 |
| Confidence Loss | 0\.003 |

 **Sources:** [boltz2\.py L898-L903](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L898-L903)

### Sequential vs Parallel Execution

 During inference, the confidence module can run in two modes:

 1. **Parallel Mode** \(default\): Processes all diffusion samples simultaneously
2. **Sequential Mode** \(`run_confidence_sequentially=True`\): Processes one sample at a time to reduce memory usage

  **Sources:** [boltz2\.py L409](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L409-L409) [boltz2\.py L603](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L603-L603)

## Training

### Loss Function

 The confidence loss is computed by the `confidence_loss` function from `boltz.model.loss.confidencev2` \(Boltz\-2\) or `boltz.model.loss.confidence` \(Boltz\-1\)\.

 **Loss Components:**

| Component | Purpose | Training Monitor |
| --- | --- | --- |
| plddt\_loss | Per\-token quality prediction | train/plddt\_loss |
| resolved\_loss | Binary resolution prediction | train/resolved\_loss |
| pde\_loss | Pairwise distance error | train/pde\_loss |
| pae\_loss | Pairwise aligned error | train/pae\_loss \(if alpha\_pae \> 0\) |

 **Sources:** [boltz2\.py L879-L893](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L879-L893) [boltz2\.py L141-L147](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L141-L147)

### Symmetry Correction

 During training, the system can optionally perform symmetry correction to handle symmetric structures \(e\.g\., homodimers\):

  This ensures the confidence model learns to predict confidence based on the optimal alignment accounting for symmetry\.

 **Sources:** [boltz2\.py L851-L862](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L851-L862)

### Training Metrics

 The model tracks various confidence\-related metrics during training:

  **Sources:** [boltz2\.py L138-L147](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L138-L147)

## Inference and Output

### Prediction Workflow

  **Sources:** [boltz2\.py L1057-L1121](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1057-L1121)

### Output Files

 The confidence prediction system generates several output files:

| File | Content | Format |
| --- | --- | --- |
| confidence\_\[name\]\_model\_N\.json | Aggregated confidence scores | JSON |
| plddt\_\[name\]\_model\_N\.npz | Per\-token pLDDT scores | NumPy array |
| pae\_\[name\]\_model\_N\.npz | PAE matrix \(if \-\-write\_full\_pae\) | NumPy array |
| pde\_\[name\]\_model\_N\.npz | PDE matrix \(if \-\-write\_full\_pde\) | NumPy array |
| \[name\]\_model\_N\.cif | Structure with pLDDT in B\-factor column | mmCIF |

 **JSON Structure:**

  **Sources:** [prediction\.md?plain=1 L198-L225](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L198-L225)

### Ranking Predictions

 When multiple diffusion samples are generated \(controlled by `--diffusion_samples` flag\), predictions are ranked by `confidence_score`:

  The top\-ranked prediction is output as `model_0.cif`, second as `model_1.cif`, etc\.

 **Sources:** [prediction\.md?plain=1 L196](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L196-L196) [boltz2\.py L1085-L1094](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1085-L1094)

## Validation Metrics

### Boltz\-1 Validation

 Boltz\-1 tracks extensive validation metrics comparing predicted confidence to true structure quality:

| Metric Type | Purpose | Example Keys |
| --- | --- | --- |
| LDDT per molecule type | Structure quality by type | val/lddt\_protein, val/lddt\_ligand\_protein |
| Distogram LDDT | Quality from distogram alone | val/disto\_lddt |
| Top\-1 LDDT | Best sample by pLDDT ranking | val/top1\_lddt |
| iPLDDT Top\-1 LDDT | Best sample by interface pLDDT | val/iplddt\_top1\_lddt |
| PDE Top\-1 LDDT | Best sample by PDE ranking | val/pde\_top1\_lddt |
| PTM/iPTM Top\-1 LDDT | Best sample by TM\-score | val/ptm\_top1\_lddt, val/iptm\_top1\_lddt |
| MAE metrics | Mean absolute error of predictions | val/MAE\_plddt\_protein, val/MAE\_pae\_protein |

 **Sources:** [boltz1\.py L880-L1064](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L880-L1064)

### Metric Computation

 During validation, several metrics are computed:

  **Sources:** [boltz1\.py L618-L878](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L618-L878)

### Compilation

 The confidence module can be compiled with PyTorch's `torch.compile` for improved performance:

  **Sources:** [boltz2\.py L316-L319](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L316-L319) [boltz1\.py L253-L256](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py#L253-L256)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction](https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction) on DeepWiki*