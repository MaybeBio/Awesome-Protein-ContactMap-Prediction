---
title: "Affinity Prediction"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.6-affinity-prediction
---
# Affinity Prediction

# Affinity Prediction

> **Relevant source files**
> - [README\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1)
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/model/models/boltz1\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz1.py)
> - [src/boltz/model/models/boltz2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py)

## Purpose and Scope

 This document describes Boltz\-2's affinity prediction capability, which jointly models complex structures and binding affinities for protein\-ligand interactions\. The affinity module predicts binding strength in two complementary ways: a regression task for ranking binders and a binary classification task for discriminating binders from non\-binders\. This functionality is exclusive to Boltz\-2 and not available in Boltz\-1\. For information about structure prediction, see [Boltz\-1 Model](https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model) and [Boltz\-2 Model](https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model)\. For confidence prediction, see [Confidence Prediction](https://deepwiki.com/jwohlwend/boltz/3.5-confidence-prediction)\.

 **Sources:** [README\.md?plain=1 L15-L18](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L15-L18) [prediction\.md?plain=1 L51-L53](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L51-L53) [prediction\.md?plain=1 L103-L104](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L103-L104)

---

## Overview

 Boltz\-2's affinity prediction produces two distinct outputs trained on different datasets with different supervision signals:

| Output | Use Case | Value Range | Supervision |
| --- | --- | --- | --- |
| affinity\_probability\_binary | Hit discovery, binder vs\. decoy discrimination | 0 to 1 \(probability\) | Binary classification |
| affinity\_pred\_value | Lead optimization, ranking active molecules | Real number \(log10\(IC50\) in μM\) | Regression on IC50 values |

 The dual\-output design reflects different stages of drug discovery: early\-stage screening requires identifying any binders, while later optimization requires precise quantification of relative binding strength among known actives\.

 **Sources:** [prediction\.md?plain=1 L240-L248](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L240-L248) [README\.md?plain=1 L51-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/README.md?plain=1#L51-L52)

---

## Architecture Integration

 The affinity module operates as a downstream task in the Boltz\-2 architecture, consuming representations from the trunk network:

  **Diagram: Affinity Module in Boltz\-2 Pipeline**

 The affinity module is invoked only during prediction, not during structure training\. It receives:

 - **s\_inputs**: Token\-level sequence embeddings from `InputEmbedder` \(re\-computed with `affinity=True`\)
- **z**: Filtered pair representation \(cross\-interface pairs only\)
- **x\_pred**: Best predicted coordinates from diffusion sampling

 **Sources:** [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721)

---

## Dual Prediction Outputs

### Affinity Value \(Regression\)

 The `affinity_pred_value` output predicts binding affinity as `log10(IC50)` where IC50 is measured in micromolar \(μM\)\. Lower values indicate stronger binding:

 - **IC50 = 10⁻⁹ M** → model outputs **\-3** \(strong binder\)
- **IC50 = 10⁻⁶ M** → model outputs **0** \(moderate binder\)
- **IC50 = 10⁻⁴ M** → model outputs **2** \(weak binder/decoy\)

 This output should only be used when comparing **active molecules**, not for discriminating actives from inactives\. It is intended for ligand optimization stages such as hit\-to\-lead and lead\-optimization\.

 **Conversion to pIC50:** The output can be converted to pIC50 in kcal/mol using: `pIC50 = (6 - log10_IC50) × 1.364`

### Binary Probability \(Classification\)

 The `affinity_probability_binary` output ranges from 0 to 1 and represents the predicted probability that the ligand is a binder\. This output is appropriate for hit\-discovery stages where the goal is to detect binders from decoys\. It uses a sigmoid activation on the model's logits:

```
probability = sigmoid(affinity_logits_binary)
```

 **Sources:** [prediction\.md?plain=1 L240-L249](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L240-L249) [boltz2\.py L640-L643](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L640-L643) [boltz2\.py L653-L656](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L653-L656)

---

## Ensemble Architecture

 Boltz\-2 supports an optional two\-model ensemble for affinity prediction, controlled by the `affinity_ensemble` flag:

  **Diagram: Ensemble Affinity Architecture**

 When `affinity_ensemble=True`, the model maintains two separate `AffinityModule` instances:

 - `self.affinity_module1` \(lines 323\-327\)
- `self.affinity_module2` \(lines 328\-332\)

 Both modules receive identical inputs and their outputs are averaged:

  The individual model predictions are also returned as `affinity_pred_value1`, `affinity_probability_binary1`, `affinity_pred_value2`, `affinity_probability_binary2` for analysis\.

 **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L321-L349) [boltz2\.py L629-L686](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L629-L686)

---

## Molecular Weight Correction

 When `affinity_mw_correction=True`, the ensemble's `affinity_pred_value` is adjusted using a linear correction based on the ligand's molecular weight:

  This correction accounts for known biases in IC50 measurements related to molecular size\. The molecular weight is transformed via a power of 0\.3 before applying the correction\.

 **Sources:** [boltz2\.py L687-L697](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L687-L697) [prediction\.md?plain=1 L158](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L158-L158)

---

## Affinity Computation Process

 The affinity prediction involves several key steps:

  **Diagram: Affinity Prediction Workflow**

### Step\-by\-Step Process

 1. **Structure Selection**: After generating multiple diffusion samples, the model selects the best structure based on the highest `iptm` \(interface predicted TM\-score\)\.
2. **Cross\-Interface Masking**: A mask is computed to retain only receptor\-ligand and ligand\-ligand pair representations:
3. **Re\-embedding**: Input features are re\-embedded with `affinity=True` to enable affinity\-specific feature processing\.
4. **Module Forward Pass**: The `AffinityModule` processes the filtered representations and coordinates to produce raw predictions\.
5. **Post\-processing**: Logits are converted to probabilities, ensemble averaging is applied, and optional MW correction is performed\.

 **Sources:** [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721)

---

## Input Requirements and Constraints

### Ligand Specifications

 Affinity prediction requires specifying the ligand chain in the YAML input:

  **Constraints:**

 - The `binder` must be a **ligand** chain \(not protein, DNA, or RNA\)
- Maximum **128 atoms** \(counting heavy atoms and hydrogens kept by `RDKit RemoveHs`\)
- Recommended maximum **56 atoms** \(training limit\)
- Only **one ligand** can be specified for affinity computation per prediction

### Target Specifications

 - Currently optimized for **protein targets** only
- While the code accepts RNA/DNA/co\-factor targets, predictions for these are unreliable
- The model masks receptor\-ligand interfaces using `mol_type == 0` \(protein\) for the receptor

 **Sources:** [prediction\.md?plain=1 L103-L104](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L103-L104) [boltz2\.py L609-L618](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L609-L618)

---

## Output Format

 Affinity predictions are saved to `affinity_[input_name].json` with the following structure:

### Field Descriptions

| Field | Description | Present When |
| --- | --- | --- |
| affinity\_pred\_value | Ensemble average predicted log10\(IC50\) | Always |
| affinity\_probability\_binary | Ensemble average binder probability | Always |
| affinity\_pred\_value1 | Model 1 predicted log10\(IC50\) | affinity\_ensemble=True |
| affinity\_probability\_binary1 | Model 1 binder probability | affinity\_ensemble=True |
| affinity\_pred\_value2 | Model 2 predicted log10\(IC50\) | affinity\_ensemble=True |
| affinity\_probability\_binary2 | Model 2 binder probability | affinity\_ensemble=True |

 When `affinity_ensemble=False`, only the main ensemble fields are present \(and represent a single model's output\)\.

 **Sources:** [prediction\.md?plain=1 L228-L238](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L228-L238) [boltz2\.py L1107-L1120](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1107-L1120)

---

## Command\-Line Options

 The following CLI flags control affinity prediction behavior:

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-affinity\_mw\_correction | flag | False | Enable molecular weight correction |
| \-\-sampling\_steps\_affinity | int | 200 | Number of sampling steps for affinity prediction |
| \-\-diffusion\_samples\_affinity | int | 5 | Number of diffusion samples for affinity |
| \-\-affinity\_checkpoint | path | None | Optional checkpoint specifically for affinity module |

 The affinity module uses the same base model checkpoint but can optionally load affinity\-specific weights\.

 **Sources:** [prediction\.md?plain=1 L158-L161](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L158-L161)

---

## Implementation Details

### Module Initialization

 In `Boltz2.__init__()`, affinity modules are conditionally instantiated:

  **Sources:** [boltz2\.py L321-L349](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L321-L349)

### Forward Pass Integration

 Affinity prediction occurs after structure generation in `Boltz2.forward()`:

 1. Check if `self.affinity_prediction` is enabled
2. Compute receptor and ligand masks from `mol_type` and `affinity_token_mask`
3. Filter pair representation to cross\-interface pairs
4. Select best structure by `iptm` ranking
5. Re\-embed inputs with `affinity=True` flag
6. Run affinity module\(s\) in `float` precision \(autocast disabled\)
7. Apply sigmoid to binary logits
8. Ensemble average if applicable
9. Apply MW correction if enabled

 **Sources:** [boltz2\.py L608-L721](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L608-L721)

### Prediction Step Output

 During prediction \(`predict_step()`\), affinity results are added to the output dictionary:

  **Sources:** [boltz2\.py L1107-L1120](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L1107-L1120)

---

## Key Hyperparameters

| Parameter | Default/Typical | Description |
| --- | --- | --- |
| affinity\_prediction | False | Enable affinity prediction module |
| affinity\_ensemble | False | Use two\-model ensemble |
| affinity\_mw\_correction | True | Apply molecular weight correction |
| compile\_affinity | False | Compile affinity module with torch\.compile |
| affinity\_model\_args | Model\-specific | Configuration for single AffinityModule |
| affinity\_model\_args1 | Model\-specific | Configuration for ensemble model 1 |
| affinity\_model\_args2 | Model\-specific | Configuration for ensemble model 2 |

 These parameters are set in the model configuration YAML files and control the instantiation and behavior of the affinity prediction system\.

 **Sources:** [boltz2\.py L59-L69](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L59-L69) [boltz2\.py L295-L297](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/models/boltz2.py#L295-L297)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.6-affinity-prediction](https://deepwiki.com/jwohlwend/boltz/3.6-affinity-prediction) on DeepWiki*