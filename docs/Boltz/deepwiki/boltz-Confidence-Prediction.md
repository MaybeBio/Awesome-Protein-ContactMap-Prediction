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
> - [src/boltz/data/write/mmcif\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/mmcif.py)
> - [src/boltz/data/write/pdb\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/pdb.py)
> - [src/boltz/data/write/writer\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/writer.py)
> - [src/boltz/model/layers/confidence\_utils\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/confidence_utils.py)
> - [src/boltz/model/loss/confidence\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py)
> - [src/boltz/model/modules/confidence\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence.py)
> - [src/boltz/model/modules/confidence\_utils\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence_utils.py)
> - [src/boltz/model/modules/confidencev2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py)
> - [src/boltz/model/modules/encoders\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/encoders.py)

## Purpose and Scope

 The Confidence Prediction system in Boltz estimates the reliability of predicted structures by computing various quality metrics\. This module predicts per\-residue and pairwise confidence scores that indicate how accurate different regions of the predicted structure are likely to be\. These metrics are essential for ranking multiple predictions and identifying well\-modeled versus uncertain regions\.

 The `ConfidenceModule` acts as a specialized head that processes representations from the trunk and coordinates from the structure module to output binned probability distributions for various error metrics\.

 **Sources:** [confidencev2\.py L19-L42](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L19-L42) [boltz2\.py L304-L319](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L304-L319)

## Architecture Overview

 The confidence prediction system operates as a post\-processing module that analyzes the model's internal representations and predicted coordinates\. It produces multiple types of confidence scores, which are then aggregated into summary statistics\.

  **Sources:** [confidencev2\.py L109-L218](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L109-L218) [confidencev2\.py L328-L360](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L328-L360)

## Confidence Module Implementation

### Boltz\-1 vs Boltz\-2 Confidence Modules

 Boltz uses two versions of the confidence module depending on the model architecture\. Boltz\-2 introduces more advanced conditioning and symmetry handling\.

| Aspect | Boltz\-1 \(ConfidenceModule\) | Boltz\-2 \(ConfidenceModule\) |
| --- | --- | --- |
| Module Path | boltz\.model\.modules\.confidence | boltz\.model\.modules\.confidencev2 |
| Positional Encoding | RelativePositionEncoder | RelativePositionEncoder \(with fix\_sym\_check\) |
| Conditioning | Basic input concatenation | ContactConditioning module |
| S\-Trunk Update | Configurable | Optional via no\_update\_s |
| Atom Rep | token\_to\_rep\_atom | token\_to\_rep\_atom |

 **Sources:** [confidence\.py L20-L40](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence.py#L20-L40) [confidencev2\.py L19-L43](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L19-L43) [confidencev2\.py L89-L93](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L89-L93)

### Feature Integration and Conditioning

 In Boltz\-2, the confidence module integrates several features before passing them to the internal `PairformerModule`:

 1. **Distance Bins**: Predicted coordinates `x_pred` are converted into distance matrices and embedded into `token_z` space using `dist_bin_pairwise_embed` [confidencev2\.py L49-L50](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L49-L50)
2. **Relative Position**: Uses a `RelativePositionEncoder` to add structural context to the pair representation [confidencev2\.py L77-L79](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L77-L79)
3. **Contact Conditioning**: The `ContactConditioning` layer incorporates spatial constraints and potential contact information into the pairwise features [confidencev2\.py L89-L93](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L89-L93)
4. **Outer Product**: Optional addition of token embedding outer products to the pair representation via `s_to_z_prod_out` [confidencev2\.py L180-L184](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L180-L184)

 **Sources:** [confidencev2\.py L165-L184](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L165-L184) [confidencev2\.py L49-L50](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L49-L50)

### Aggregated Metrics Calculation

 The model converts raw logits into interpretable scores using `compute_aggregated_metric` and `compute_ptms`\.

| Function | Logic |
| --- | --- |
| compute\_aggregated\_metric | Computes the expected value across bins using a softmax weighted sum of bin center values src/boltz/model/modules/confidence\_utils\.py8\-34 |
| tm\_function | Implements the TM\-score rescaling: $1 / \(1 \+ \(d/d\_0\)^2\)$ where $d\_0$ depends on sequence length src/boltz/model/modules/confidence\_utils\.py37\-54 |
| compute\_ptms | Masks collinear/overlapping tokens and computes complex\-wide and interface\-specific scores \(pTM, ipTM\) src/boltz/model/modules/confidence\_utils\.py57\-123 |

 **Sources:** [confidence\_utils\.py L8-L123](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence_utils.py#L8-L123) [confidence\_utils\.py L115-L173](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/confidence_utils.py#L115-L173)

## Confidence Metrics Interpretation

### pLDDT \(Predicted Local Distance Difference Test\)

 Per\-token metric estimating local accuracy\. In the output mmCIF or PDB files, this value is stored in the B\-factor column, scaled to \[0, 100\]\.

 - **Implementation**: Predicted by `plddt_head` in `ConfidenceHeads` [confidencev2\.py L355](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L355-L355)
- **Output Writing**: `to_pdb` and `to_mmcif` functions multiply the \[0, 1\] pLDDT by 100 for the B\-factor field [pdb\.py L108-L111](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/pdb.py#L108-L111) [mmcif\.py L206-L209](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/mmcif.py#L206-L209)

 **Sources:** [pdb\.py L108-L111](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/pdb.py#L108-L111) [mmcif\.py L206-L209](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/mmcif.py#L206-L209) [confidencev2\.py L355](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L355-L355)

### PAE and PDE

 **PAE \(Predicted Aligned Error\)** and **PDE \(Predicted Distance Error\)** provide pairwise error estimates\.

 - **PAE**: Used to derive `ptm` and `iptm`\. It measures the error in position of token $i$ if the predicted and true structures are aligned on token $j$\.
- **PDE**: Measures the absolute error in the distance between token $i$ and token $j$\.
- **Ligand/Protein Specificity**: `compute_ptms` uses `mol_type` features to calculate specific interface scores like `ligand_iptm` and `protein_iptm` [confidence\_utils\.py L125-L158](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence_utils.py#L125-L158)

 **Sources:** [confidence\_utils\.py L125-L158](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence_utils.py#L125-L158) [confidencev2\.py L356-L357](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidencev2.py#L356-L357)

### Ranking Logic

 The `BoltzWriter` uses the `confidence_score` to rank multiple samples generated during diffusion\.

  The final `confidence_score` is a weighted combination: $0\.8 \\times \\text\{complex\_plddt\} \+ 0\.2 \\times \\text\{ipTM\}$ \(or pTM if ipTM is unavailable\)\.

 **Sources:** [writer\.py L74-L76](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/write/writer.py#L74-L76) [boltz2\.py L1085-L1094](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/models/boltz2.py#L1085-L1094)

## Training and Loss Functions

 The confidence module is trained using a multi\-component loss function implemented in `boltz.model.loss.confidence`\.

### Loss Components

  - **plddt\_loss**: Calculated by comparing predicted distances `pred_d` to true distances `true_d` within a 15Å \(protein\) or 30Å \(nucleotide\) cutoff [confidence\.py L192-L215](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L192-L215)
- **pde\_loss**: Binned cross\-entropy loss on the absolute difference between predicted and true pairwise distances [confidence\.py L242-L280](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L242-L280)
- **resolved\_loss**: Binary cross\-entropy predicting if an atom's position is resolved in the ground truth [confidence\.py L87-L133](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L87-L133)

 **Sources:** [confidence\.py L7-L84](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L7-L84) [confidence\.py L192-L215](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L192-L215) [confidence\.py L242-L280](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/loss/confidence.py#L242-L280)

### Frame Prediction for PAE

 To compute PAE, the model must predict local reference frames for every token\. For non\-polymers \(ligands\) with at least 3 atoms, the model dynamically constructs frames based on the nearest neighbors in the predicted coordinates [confidence\_utils\.py L34-L95](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/confidence_utils.py#L34-L95)

 **Sources:** [confidence\_utils\.py L34-L95](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/layers/confidence_utils.py#L34-L95) [confidence\_utils\.py L84-L86](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/model/modules/confidence_utils.py#L84-L86)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction](https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction) on DeepWiki*