---
title: "Loss Functions"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/5.6-loss-functions
---
# Loss Functions

# Loss Functions

> **Relevant source files**
> - [openfold/data/data\_modules\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/data/data_modules.py)
> - [openfold/utils/loss\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py)
> - [train\_openfold\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py)

## Purpose and Scope

 This document explains the loss functions used to train the OpenFold model\. The loss system aggregates multiple complementary objectives that guide the model to predict accurate protein structures\. For information about the model architecture that these losses train, see [Model Architecture](https://deepwiki.com/aqlaboratory/openfold/5-model-architecture)\. For training pipeline details, see [Training Pipeline](https://deepwiki.com/aqlaboratory/openfold/4.1-training-pipeline)\.

 All loss function implementations are located in [openfold/utils/loss\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py) The main entry point is the `AlphaFoldLoss` class, which combines individual loss components with configurable weights\.

---

## Loss System Architecture

 The OpenFold training loss consists of multiple components that collectively optimize different aspects of structure prediction:

```mermaid
flowchart TD

SM["StructureModule outputs<br>sm['frames'], sm['positions'],<br>sm['angles']"]
LOGITS["Auxiliary head outputs<br>distogram_logits, lddt_logits,<br>tm_logits, masked_msa_logits"]
FINAL["final_atom_positions"]
GT_POS["backbone_rigid_tensor<br>all_atom_positions"]
GT_ANGLES["chi_angles_sin_cos<br>chi_mask"]
GT_MSA["true_msa<br>bert_mask"]
GT_OTHER["seq_mask, aatype,<br>residue_index, asym_id"]
AGGREGATOR["loss() method<br>Computes & weights all losses"]
FAPE["fape_loss<br>Backbone + Sidechain FAPE"]
LDDT["lddt_loss<br>pLDDT prediction"]
DIST["distogram_loss<br>Distance distribution"]
TM["tm_loss<br>TM-score prediction"]
CHI["supervised_chi_loss<br>Torsion angles"]
VIOL["violation_loss<br>Structural violations"]
MSA["masked_msa_loss<br>BERT-style MSA"]
EXP["experimentally_resolved_loss<br>Atom resolution"]
COM["chain_center_of_mass_loss<br>Multimer only"]
TOTAL["Total Loss<br>Scaled by sqrt(min(seq_len, crop_len))"]

SM --> FAPE
SM --> CHI
SM --> VIOL
LOGITS --> LDDT
LOGITS --> DIST
LOGITS --> TM
LOGITS --> MSA
LOGITS --> EXP
FINAL --> FAPE
FINAL --> LDDT
FINAL --> TM
FINAL --> VIOL
FINAL --> COM
GT_POS --> FAPE
GT_POS --> LDDT
GT_POS --> DIST
GT_POS --> TM
GT_POS --> VIOL
GT_POS --> COM
GT_ANGLES --> CHI
GT_MSA --> MSA
GT_OTHER --> FAPE
GT_OTHER --> LDDT
GT_OTHER --> DIST
GT_OTHER --> TM
GT_OTHER --> CHI
GT_OTHER --> VIOL
GT_OTHER --> MSA
GT_OTHER --> EXP
GT_OTHER --> COM
FAPE --> AGGREGATOR
LDDT --> AGGREGATOR
DIST --> AGGREGATOR
TM --> AGGREGATOR
CHI --> AGGREGATOR
VIOL --> AGGREGATOR
MSA --> AGGREGATOR
EXP --> AGGREGATOR
COM --> AGGREGATOR
AGGREGATOR --> TOTAL

subgraph subGraph3 ["Loss Components"]
    FAPE
    LDDT
    DIST
    TM
    CHI
    VIOL
    MSA
    EXP
    COM
end

subgraph subGraph2 ["AlphaFoldLoss Class"]
    AGGREGATOR
end

subgraph subGraph1 ["Ground Truth (batch)"]
    GT_POS
    GT_ANGLES
    GT_MSA
    GT_OTHER
end

subgraph subGraph0 ["Model Outputs"]
    SM
    LOGITS
    FINAL
end
```

 **Sources:** [loss\.py L1685-L1793](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1685-L1793) [train\_openfold\.py L52](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L52-L52) [train\_openfold\.py L115-L117](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L115-L117)

---

## AlphaFoldLoss Class

 The `AlphaFoldLoss` class \([loss\.py L1685-L1793](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1685-L1793)\) aggregates all individual loss components into a single training objective\.

### Key Methods

| Method | Purpose |
| --- | --- |
| \_\_init\_\_\(config\) | Initializes with configuration specifying weights and parameters for each loss component |
| loss\(out, batch, \_return\_breakdown\) | Computes all enabled losses and returns weighted sum |
| forward\(out, batch, \_return\_breakdown\) | Wrapper around loss\(\) for standard PyTorch Module interface |

### Loss Computation Flow

```mermaid
flowchart TD

INPUT["Model outputs + batch"]
PREPROC["Preprocessing"]
VIOL_CHECK["Violations<br>computed?"]
COMPUTE_VIOL["find_structural_violations()"]
RENAME_CHECK["Ground truth<br>renamed?"]
COMPUTE_RENAME["compute_renamed_ground_truth()"]
LOSS_FNS["Compute individual losses"]
WEIGHT["Apply config weights"]
SUM["Sum weighted losses"]
SCALE["Scale by sqrt(min(seq_len, crop_len))"]
OUTPUT["Return total loss + breakdown"]

INPUT --> PREPROC
PREPROC --> VIOL_CHECK
VIOL_CHECK -->|"No"| COMPUTE_VIOL
VIOL_CHECK -->|"Yes"| RENAME_CHECK
COMPUTE_VIOL -->|"Yes"| RENAME_CHECK
RENAME_CHECK -->|"No"| COMPUTE_RENAME
RENAME_CHECK -->|"Yes"| LOSS_FNS
COMPUTE_RENAME -->|"Yes"| LOSS_FNS
LOSS_FNS --> WEIGHT
WEIGHT --> SUM
SUM --> SCALE
SCALE --> OUTPUT
```

### Usage in Training

 The loss is instantiated in `OpenFoldWrapper` \([train\_openfold\.py L45-L60](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L45-L60)\):

```
self.loss = AlphaFoldLoss(config.loss)
```

 During training \([train\_openfold\.py L115-L117](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L115-L117)\):

```
loss, loss_breakdown = self.loss(    outputs, batch, _return_breakdown=True)
```

 **Sources:** [loss\.py L1685-L1793](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1685-L1793) [train\_openfold\.py L45-L60](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L45-L60) [train\_openfold\.py L115-L117](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L115-L117)

---

## FAPE Loss \(Frame Aligned Point Error\)

 FAPE is the primary geometric loss that measures structural accuracy by comparing predicted and ground truth atomic positions in local reference frames\. This makes the loss invariant to global rotations and translations\.

### Components

 FAPE loss has two main components:

 1. **Backbone FAPE**: Measures accuracy of backbone rigid transformations
2. **Sidechain FAPE**: Measures accuracy of all\-atom sidechain positions

### Backbone FAPE

 The `backbone_loss` function \([loss\.py L169-L232](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L169-L232)\) computes FAPE for backbone frames:

```mermaid
flowchart TD

TRAJ["Trajectory of backbone frames<br>traj: [*, N_recycle, N_res, 7 or 4x4]"]
GT_FRAMES["Ground truth backbone frames<br>backbone_rigid_tensor: [*, N_res, 4, 4]"]
CONVERT["Convert to Rigid objects<br>from_tensor_7() or from_tensor_4x4()"]
CONVERT_GT["Convert to Rigid objects<br>from_tensor_4x4()"]
TRANSFORM["Transform points to local frames<br>invert()[..., None].apply(positions)"]
TRANSFORM_GT["Transform points to local frames<br>invert()[..., None].apply(positions)"]
DISTANCE["Compute L2 distances<br>sqrt(sum((pred - gt)^2))"]
CLAMP["Clamping<br>enabled?"]
CLAMP_DIST["clamp(dist, max=clamp_distance)"]
NORM["Normalize by length_scale"]
MASK["Apply masks<br>frames_mask, positions_mask"]
MULTIMER["Multimer<br>mode?"]
SEPARATE["Compute intra-chain and<br>interface losses separately"]
MEAN["Compute mean FAPE"]
WEIGHT_MULTI["Apply intra_chain_backbone.weight<br>and interface_backbone.weight"]
FINAL_FAPE["Final backbone FAPE loss"]

TRAJ --> CONVERT
GT_FRAMES --> CONVERT_GT
CONVERT --> TRANSFORM
CONVERT_GT --> TRANSFORM_GT
TRANSFORM --> DISTANCE
TRANSFORM_GT --> DISTANCE
DISTANCE --> CLAMP
CLAMP -->|"Yes"| CLAMP_DIST
CLAMP -->|"No"| NORM
CLAMP_DIST -->|"Yes"| NORM
NORM --> MASK
MASK --> MULTIMER
MULTIMER -->|"Yes"| SEPARATE
MULTIMER -->|"No"| MEAN
SEPARATE --> WEIGHT_MULTI
WEIGHT_MULTI --> FINAL_FAPE
MEAN --> FINAL_FAPE
```

 **Key parameters** \(from config\):

 - `length_scale`: Normalizes distances \(typically 10\.0 Angstroms\)
- `clamp_distance`: Maximum distance error to consider \(typically 10\.0\)
- `use_clamped_fape`: Whether to use clamping \(can be toggled during training\)

 For multimer mode \([loss\.py L293-L306](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L293-L306)\), the loss separates intra\-chain and interface regions:

 - **Intra\-chain**: Atoms within the same chain \(same `asym_id`\)
- **Interface**: Atoms across different chains

 **Sources:** [loss\.py L82-L166](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L82-L166) [loss\.py L169-L232](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L169-L232) [loss\.py L286-L325](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L286-L325)

### Sidechain FAPE

 The `sidechain_loss` function \([loss\.py L235-L283](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L235-L283)\) computes FAPE for all 8 rigid groups per residue \(backbone \+ 7 sidechain torsion frames\):

 1. Handles alternative ground truth for symmetric sidechains \(e\.g\., Phe, Tyr\)
2. Uses the final predicted sidechain frames from `sm["sidechain_frames"][-1]`
3. Compares all atom14 positions in their local rigid group frames

 **Sources:** [loss\.py L235-L283](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L235-L283)

### FAPE Algorithm

 The core FAPE computation \([loss\.py L82-L166](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L82-L166)\):

 $$\\text\{FAPE\} = \\frac\{1\}\{N\_\{\\text\{frames\}\} \\cdot N\_\{\\text\{pts\}\}\} \\sum\_\{f,p\} \\text\{mask\}\_\{f,p\} \\cdot \\frac\{\| T\_f^\{\-1\}\(\\mathbf\{x\}\_p^\{\\text\{pred\}\}\) \- T\_f^\{\-1\}\(\\mathbf\{x\}*p^\{\\text\{gt\}\}\) \|\}\{d*\{\\text\{scale\}\}\}$$

 Where:

 - $T\_f^\{\-1\}$ inverts frame $f$ and transforms points to its local coordinates
- $\\mathbf\{x\}\_p$ are 3D positions of atoms
- $d\_\{\\text\{scale\}\}$ is the length scale parameter
- Optional L1 clamping at `l1_clamp_distance`

 **Sources:** [loss\.py L82-L166](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L82-L166)

---

## LDDT Loss \(pLDDT Prediction\)

 The LDDT loss trains the model to predict per\-residue local distance difference test \(lDDT\) scores, which serve as confidence estimates\.

### Computation

```mermaid
flowchart TD

PRED_POS["Predicted CA positions<br>all_atom_pred_pos[..., CA, :]"]
GT_POS["Ground truth CA positions<br>all_atom_positions[..., CA, :]"]
LOGITS["pLDDT logits<br>lddt_logits: [*, N_res, 50]"]
DIST_PRED["Compute pairwise distances<br>dmat_pred: [*, N_res, N_res]"]
DIST_GT["Compute pairwise distances<br>dmat_true: [*, N_res, N_res]"]
DIFF["Distance differences<br>|dmat_pred - dmat_true|"]
SCORE["Count differences < thresholds<br>0.5, 1.0, 2.0, 4.0 Angstroms"]
LDDT_VAL["Average to get lDDT<br>score: [*, N_res]"]
DETACH["Detach from graph<br>(target, not differentiable)"]
BIN["Convert to bin indices<br>bin_index = floor(score * 50)"]
ONEHOT["One-hot encode<br>lddt_ca_one_hot: [*, N_res, 50]"]
CE["Softmax cross-entropy<br>with one-hot target"]
MASK_LOSS["Mask by all_atom_mask<br>and resolution filter"]
FINAL["Mean LDDT loss"]

PRED_POS --> DIST_PRED
GT_POS --> DIST_GT
DIST_PRED --> DIFF
DIST_GT --> DIFF
DIFF --> SCORE
SCORE --> LDDT_VAL
LDDT_VAL --> DETACH
DETACH --> BIN
BIN --> ONEHOT
LOGITS --> CE
ONEHOT --> CE
CE --> MASK_LOSS
MASK_LOSS --> FINAL
```

### Key Features

 1. **Local metric**: Only considers atom pairs within 15 Angstroms cutoff
2. **Multi\-threshold scoring**: Uses 4 distance thresholds \(0\.5, 1\.0, 2\.0, 4\.0 Å\)
3. **Binned prediction**: Model predicts distribution over 50 bins from 0\-100
4. **Resolution filtering**: Only applied to structures with resolution between 0\.1\-3\.0 Å

 **Implementation:** [loss\.py L428-L481](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L428-L481) \(`lddt`\), [loss\.py L484-L504](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L484-L504) \(`lddt_ca`\), [loss\.py L507-L559](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L507-L559) \(`lddt_loss`\)

 **Sources:** [loss\.py L428-L481](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L428-L481) [loss\.py L484-L504](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L484-L504) [loss\.py L507-L559](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L507-L559)

---

## Distogram Loss

 The distogram loss trains the model to predict the distribution of C\-beta \(or C\-alpha for glycine\) pairwise distances\.

### Implementation

 The `distogram_loss` function \([loss\.py L562-L607](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L562-L607)\):

 1. **Compute pairwise distances** between pseudo\-beta atoms:   ``` dists = sum((pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :])^2) ```
2. **Bin distances**: 64 bins with boundaries from 2\.3125² to 21\.6875² Ų
3. **Cross\-entropy loss**: Between predicted logits and true distance bins
4. **Masking**: Applied via `pseudo_beta_mask` to exclude missing residues

 The distogram provides a differentiable distance restraint that helps the model learn distance geometry principles\.

 **Sources:** [loss\.py L562-L607](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L562-L607)

---

## TM Score Loss

 For multimer and some training stages, the model predicts the expected TM\-score \(Template Modeling score\) between predicted and true structures\.

### Computation Flow

```mermaid
flowchart TD

PRED_AFF["Predicted backbone affines<br>final_affine_tensor"]
GT_AFF["Ground truth affines<br>backbone_rigid_tensor"]
LOGITS["TM logits<br>tm_logits: [*, N_res, N_res, 64]"]
INVERT_PRED["Invert & apply to translations<br>get local frame coordinates"]
INVERT_GT["Invert & apply to translations<br>get local frame coordinates"]
SQ_DIFF["Squared distances<br>sum((pred - gt)^2)"]
DETACH["Detach (target)"]
BIN_TM["Bin squared distances<br>64 bins from 0 to 31^2"]
ONEHOT_TM["One-hot encode true bins"]
CE_TM["Softmax cross-entropy"]
MASK_TM["Mask by backbone_rigid_mask<br>and resolution filter"]
FINAL_TM["Mean TM loss"]

PRED_AFF --> INVERT_PRED
GT_AFF --> INVERT_GT
INVERT_PRED --> SQ_DIFF
INVERT_GT --> SQ_DIFF
SQ_DIFF --> DETACH
DETACH --> BIN_TM
BIN_TM --> ONEHOT_TM
LOGITS --> CE_TM
ONEHOT_TM --> CE_TM
CE_TM --> MASK_TM
MASK_TM --> FINAL_TM
```

### TM Score Computation

 The model can also compute predicted TM\-scores from logits using `compute_tm` \([loss\.py L670-L718](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L670-L718)\):

 $$\\text\{TM\} = \\frac\{1\}\{N\} \\sum\_\{i,j\} \\frac\{1\}\{1 \+ \(d\_\{ij\} / d\_0\)^2\}$$

 Where:

 - $d\_0 = 1\.24 \\sqrt\[3\]\{N \- 15\} \- 1\.8$ \(normalization constant\)
- Supports both intra\-chain and interface TM\-scores \(controlled by `asym_id`\)

 **Sources:** [loss\.py L670-L718](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L670-L718) [loss\.py L721-L779](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L721-L779)

---

## Supervised Chi Loss \(Torsion Angles\)

 The supervised chi loss trains the model to predict accurate sidechain torsion angles\.

### Implementation

 The `supervised_chi_loss` function \([loss\.py L328-L411](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L328-L411)\):

 1. **Extract chi angles**: From `sm["angles"]` \(predicted\) and `chi_angles_sin_cos` \(ground truth\)  - Only considers chi angles \(indices 3\-7\), skipping backbone angles
2. **Handle periodicity**: Some chi angles are π\-periodic \(e\.g\., Phe, Tyr\)  - Compares both the angle and angle\+π, taking the minimum error
3. **Angle normalization**: Penalizes unnormalized angle predictions  - `angle_norm_loss = |norm(angle_vector) - 1.0|`
4. **Masking**: Uses `chi_mask` to only supervise chi angles that exist for each residue type

 **Loss formulation:**

```
loss = chi_weight * chi_loss + angle_norm_weight * norm_loss
```

 Typical weights: `chi_weight=0.5`, `angle_norm_weight=0.02`

 **Sources:** [loss\.py L328-L411](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L328-L411)

---

## Violation Losses

 Violation losses penalize physically implausible structures by enforcing geometric constraints\.

### Overview

```mermaid
flowchart TD

PRED_POS["Predicted atom14 positions<br>sm['positions'][-1]"]
BATCH["Batch data<br>residue_index, aatype,<br>atom14_atom_exists"]
BOND["between_residue_bond_loss<br>C-N bond length & angles"]
CLASH_INTER["between_residue_clash_loss<br>Inter-residue clashes"]
CLASH_INTRA["within_residue_violations<br>Intra-residue clashes & bonds"]
BONDS_LOSS["bonds_c_n_loss_mean"]
ANGLES_LOSS["angles_ca_c_n_loss_mean<br>angles_c_n_ca_loss_mean"]
INTER_CLASH["clashes_mean_loss"]
INTRA_CLASH["per_atom_loss_sum"]
AGGREGATE["violation_loss aggregator"]
FINAL_VIOL["Total violation loss"]

PRED_POS --> BOND
PRED_POS --> CLASH_INTER
PRED_POS --> CLASH_INTRA
BATCH --> BOND
BATCH --> CLASH_INTER
BATCH --> CLASH_INTRA
BOND --> BONDS_LOSS
BOND --> ANGLES_LOSS
CLASH_INTER --> INTER_CLASH
CLASH_INTRA --> INTRA_CLASH
BONDS_LOSS --> AGGREGATE
ANGLES_LOSS --> AGGREGATE
INTER_CLASH --> AGGREGATE
INTRA_CLASH --> AGGREGATE
AGGREGATE --> FINAL_VIOL
```

### Between\-Residue Bond Loss

 The `between_residue_bond_loss` function \([loss\.py L782-L939](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L782-L939)\) enforces peptide bond geometry:

 **Constraints checked:**

 1. **C\-N bond length**: ~1\.329 Å \(1\.341 Å for proline\)
2. **CA\-C\-N angle**: Bond angle around C atom
3. **C\-N\-CA angle**: Bond angle around N atom

 **Tolerance:** Uses `tolerance_factor_soft` and `tolerance_factor_hard` \(both typically 12\.0\)

 - Soft: Applied with ReLU to compute loss
- Hard: Used to flag violations

 **Sources:** [loss\.py L782-L939](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L782-L939)

### Between\-Residue Clash Loss

 The `between_residue_clash_loss` function \([loss\.py L942-L1093](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L942-L1093)\) penalizes steric clashes:

 1. **Van der Waals radii**: Uses element\-specific radii from `residue_constants`
2. **Distance check**: Flags clashes when atoms closer than sum of radii minus tolerance
3. **Exclusions**: - Consecutive C\-N bonds \(not clashes\) - Disulfide bonds \(CYS SG\-SG\) - Same residue atoms \(handled separately\)
4. **Multimer support**: Uses `asym_id` to correctly identify consecutive residues across chains

 **Sources:** [loss\.py L942-L1093](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L942-L1093)

### Within\-Residue Violations

 The `within_residue_violations` function \([loss\.py L1096-L1183](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1096-L1183)\) checks intra\-residue geometry:

 1. **Distance bounds**: Uses `residue_constants.make_atom14_dists_bounds()` - Lower and upper bounds for all atom pairs within each residue
2. **Violations**: Penalizes distances outside bounds
3. **Per\-atom aggregation**: Sums violations for each atom

 **Sources:** [loss\.py L1096-L1183](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1096-L1183)

### Violation Loss Aggregation

 The `violation_loss` function \([loss\.py L1432-L1460](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1432-L1460)\) combines all violation components:

```
loss = (    bonds_c_n_loss_mean +    angles_ca_c_n_loss_mean +    angles_c_n_ca_loss_mean +    l_clash  # Combined inter- and intra-residue clashes)
```

 With optional averaging over number of clashes if `average_clashes=True`\.

 **Sources:** [loss\.py L1432-L1460](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1432-L1460)

---

## Masked MSA Loss

 The masked MSA loss implements BERT\-style pretraining on MSA sequences\.

### Implementation

 The `masked_msa_loss` function \([loss\.py L1595-L1625](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1595-L1625)\):

 1. **Input**:  - Predicted residue distributions: `logits` \[\*, N\_seq, N\_res, 23\] - True MSA: `true_msa` \[\*, N\_seq, N\_res\] - Mask indicating masked positions: `bert_mask` \[\*, N\_seq, N\_res\]
2. **Loss computation**:  - Softmax cross\-entropy between predicted and true amino acids - Only at masked positions \(where `bert_mask=1`\)
3. **Purpose**: Helps the model learn evolutionary patterns from MSAs

 **Sources:** [loss\.py L1595-L1625](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1595-L1625)

---

## Experimentally Resolved Loss

 The `experimentally_resolved_loss` function \([loss\.py L1571-L1592](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1571-L1592)\) trains the model to predict which atoms are experimentally resolved \(present in the crystal structure\)\.

 **Implementation:**

 1. Binary classification per atom \(resolved vs\. unresolved\)
2. Uses sigmoid cross\-entropy loss
3. Filtered by resolution range \(typically 0\.1\-3\.0 Å\)

 This helps the model learn to predict confidence in individual atom positions\.

 **Sources:** [loss\.py L1571-L1592](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1571-L1592)

---

## Chain Center of Mass Loss \(Multimer\)

 For multimer predictions, the `chain_center_of_mass_loss` function \([loss\.py L1628-L1682](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1628-L1682)\) enforces consistency of inter\-chain distances\.

### Algorithm

```mermaid
flowchart TD

PRED_CA["Predicted CA positions"]
GT_CA["Ground truth CA positions"]
ASYM["Chain IDs (asym_id)"]
COM_PRED["Compute per-chain<br>center of mass"]
COM_GT["Compute per-chain<br>center of mass"]
DIST_PRED["Pairwise chain distances<br>euclidean_distance()"]
DIST_GT["Pairwise chain distances<br>euclidean_distance()"]
DIFF["Distance differences<br>(pred - gt - clamp_distance)"]
CLAMP["Clamp at max=0<br>Only penalize distances too short"]
SQUARE["Square the error"]
WEIGHT_COM["Multiply by weight (0.05)"]
FINAL_COM["Chain COM loss"]

PRED_CA --> COM_PRED
GT_CA --> COM_GT
ASYM --> COM_PRED
ASYM --> COM_GT
COM_PRED --> DIST_PRED
COM_GT --> DIST_GT
DIST_PRED --> DIFF
DIST_GT --> DIFF
DIFF --> CLAMP
CLAMP --> SQUARE
SQUARE --> WEIGHT_COM
WEIGHT_COM --> FINAL_COM
```

 **Purpose:** Prevents chains from drifting apart or collapsing together during multimer training\.

 **Default parameters:**

 - `clamp_distance`: \-4\.0 \(negative = allow some compression\)
- `weight`: 0\.05

 **Sources:** [loss\.py L1628-L1682](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1628-L1682)

---

## Loss Configuration and Weights

 Each loss component has configurable parameters set in `model_config` \(see [Configuration System](https://deepwiki.com/aqlaboratory/openfold/5.1-configuration-system)\)\.

### Typical Loss Weights

| Loss Component | Config Key | Typical Weight |
| --- | --- | --- |
| FAPE \(backbone\) | fape\.backbone\.weight | 0\.5 |
| FAPE \(sidechain\) | fape\.sidechain\.weight | 0\.5 |
| LDDT | plddt\_loss\.weight | 0\.01 |
| Distogram | distogram\.weight | 0\.3 |
| TM score | tm\.weight | 0\.0 \(multimer: varies\) |
| Supervised chi | supervised\_chi\.weight | 1\.0 |
| Violations | violation\.weight | 0\.0 \(enabled later in training\) |
| Masked MSA | masked\_msa\.weight | 2\.0 |
| Experimentally resolved | experimentally\_resolved\.weight | 0\.01 |
| Chain COM | chain\_center\_of\_mass\.weight | 0\.05 |

### Loss Scaling

 The final loss is scaled by $\\sqrt\{\\min\(L\_\{\\text\{seq\}\}, L\_\{\\text\{crop\}\}\)\}$ where:

 - $L\_\{\\text\{seq\}\}$ is the sequence length
- $L\_\{\\text\{crop\}\}$ is the crop size

 This normalization \([loss\.py L1776-L1778](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1776-L1778)\):

```
seq_len = torch.mean(batch["seq_length"].float())crop_len = batch["aatype"].shape[-1]cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))
```

 Ensures that losses for different protein sizes are comparable\.

 **Sources:** [loss\.py L1712-L1780](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1712-L1780)

---

## Auxiliary Functions

### Ground Truth Renaming

 The `compute_renamed_ground_truth` function \([loss\.py L1463-L1568](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1463-L1568)\) handles symmetric sidechains:

 **Problem:** Some amino acids have symmetric atoms \(e\.g\., Phe ring can flip\) **Solution:** Compare predicted positions to both original and "flipped" ground truth, use the better match

 **Process:**

 1. Compute LDDT\-like score for original and alternative ground truth
2. Choose alternative if it gives better score
3. Return renamed ground truth for use in all losses

 This prevents the model from being penalized for predicting valid symmetric configurations\.

 **Sources:** [loss\.py L1463-L1568](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1463-L1568)

### Structural Violations Detection

 The `find_structural_violations` function \([loss\.py L1186-L1316](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1186-L1316)\) computes all violation metrics:

 **Outputs:**

```
{    "between_residues": {        "bonds_c_n_loss_mean": ...,        "angles_ca_c_n_loss_mean": ...,        "clashes_mean_loss": ...,        ...    },    "within_residues": {        "per_atom_loss_sum": ...,        ...    },    "total_per_residue_violations_mask": ...}
```

 Used both for computing violation loss and for masking LDDT predictions\.

 **Sources:** [loss\.py L1186-L1316](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1186-L1316)

---

## Loss in Training Loop

### Integration with OpenFoldWrapper

 The loss is used in the training step \([train\_openfold\.py L97-L122](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L97-L122)\):

```mermaid
flowchart TD

BATCH["Input batch"]
FORWARD["model(batch)"]
OUTPUTS["Model outputs"]
MULTIMER["Multimer<br>mode?"]
PERMUTE["multi_chain_permutation_align<br>Find best chain ordering"]
COMPUTE_LOSS["loss(outputs, batch)"]
BREAKDOWN["Loss breakdown dict"]
LOG["Log to wandb/tensorboard"]
BACKWARD["loss.backward()"]

BATCH --> FORWARD
FORWARD --> OUTPUTS
OUTPUTS --> MULTIMER
MULTIMER -->|"Yes"| PERMUTE
MULTIMER -->|"No"| COMPUTE_LOSS
PERMUTE --> COMPUTE_LOSS
COMPUTE_LOSS --> BREAKDOWN
BREAKDOWN --> LOG
BREAKDOWN --> BACKWARD
```

### EMA and Validation

 During validation \([train\_openfold\.py L127-L156](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L127-L156)\):

 1. EMA weights are loaded for inference
2. Loss is computed with `use_clamped_fape=0.0` \(unclamped FAPE only\)
3. Additional metrics computed: `lddt_ca`, `drmsd_ca`, `gdt_ts`, `gdt_ha`
4. Original weights restored after validation

 **Sources:** [train\_openfold\.py L97-L122](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L97-L122) [train\_openfold\.py L127-L156](https://github.com/aqlaboratory/openfold/blob/56da08ec/train_openfold.py#L127-L156)

---

## Summary Table

| Loss | Purpose | Key Parameters | Output Shape |
| --- | --- | --- | --- |
| FAPE | Geometric accuracy of frames & atoms | length\_scale, clamp\_distance | Scalar |
| LDDT | Train confidence prediction | cutoff=15\.0, no\_bins=50 | Scalar |
| Distogram | Distance distribution | min\_bin, max\_bin, no\_bins=64 | Scalar |
| TM Score | Predicted TM\-score | max\_bin=31, no\_bins=64 | Scalar |
| Supervised Chi | Torsion angle accuracy | chi\_weight, angle\_norm\_weight | Scalar |
| Violations | Physical plausibility | tolerance\_factor, clash\_overlap\_tolerance | Scalar |
| Masked MSA | MSA representation learning | num\_classes=23 | Scalar |
| Exp\. Resolved | Atom resolution prediction | min\_resolution, max\_resolution | Scalar |
| Chain COM | Multimer chain spacing | clamp\_distance=\-4\.0, weight=0\.05 | Scalar |

 All losses return scalars \(averaged over batch\) and are combined with configurable weights in `AlphaFoldLoss`\.

 **Sources:** [loss\.py L1-L1793](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/loss.py#L1-L1793)

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/5.6-loss-functions](https://deepwiki.com/aqlaboratory/openfold/5.6-loss-functions) on DeepWiki*