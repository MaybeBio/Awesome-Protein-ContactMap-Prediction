---
title: "Glossary"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/10-glossary
---
# Glossary

# Glossary

> **Relevant source files**
> - [README\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1)
> - [protenix/data/core/parser\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/core/parser.py)
> - [protenix/data/inference/json\_to\_feature\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/json_to_feature.py)
> - [protenix/data/utils\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/utils.py)
> - [protenix/model/modules/fused\_ops\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/fused_ops.py)
> - [protenix/model/modules/pairformer\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/pairformer.py)
> - [protenix/tfg/engine\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/tfg/engine.py)
> - [runner/batch\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py)
> - [scripts/prepare\_training\_data\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/prepare_training_data.py)
> - [tests/test\_fused\_dropout\_add\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/tests/test_fused_dropout_add.py)

 This glossary provides definitions for codebase\-specific terms, structural biology jargon, and architectural concepts used within Protenix\. It serves as a technical reference for onboarding engineers to navigate the relationship between AlphaFold3\-based concepts and their specific implementations in the Protenix repository\.

## 1\. Structural Biology & Data Terms

 Definitions of terms related to molecular representation and the data pipeline\.

| Term | Definition | Code Pointer |
| --- | --- | --- |
| AtomArray | A core data structure from the biotite library used throughout Protenix to store atomic coordinates and annotations \(residue names, chain IDs, etc\.\)\. | protenix/data/utils\.py24\-27 |
| Bioassembly | The functional form of a biological molecule, often consisting of multiple chains \(asymmetric units\) arranged by symmetry operations\. | scripts/prepare\_training\_data\.py29\-34 |
| CCD \(Chemical Component Dictionary\) | A reference dictionary containing standard geometric and chemical information for residues and ligands\. | protenix/data/core/ccd\.py1\-20 |
| MSA \(Multiple Sequence Alignment\) | A collection of evolutionarily related sequences used as input to provide co\-evolutionary signals for structure prediction\. | runner/msa\_search\.py1\-30 |
| Polymer Entity | A molecular chain consisting of repeating units \(amino acids for proteins, nucleotides for DNA/RNA\)\. | protenix/data/core/parser\.py66\-67 |
| Asym ID \(label\_asym\_id\) | A unique identifier for each distinct chain in a molecular structure\. | protenix/data/utils\.py112\-126 |

 **Sources:**

 - `protenix/data/utils.py` [24\-126](https://github.com/bytedance/Protenix/blob/c3bfc365/24-126)
- `protenix/data/core/parser.py` [66\-67](https://github.com/bytedance/Protenix/blob/c3bfc365/66-67)
- `scripts/prepare_training_data.py` [29\-34](https://github.com/bytedance/Protenix/blob/c3bfc365/29-34)

---

## 2\. Model Architecture Terms

 Terms defining the neural network components and the structural generation process\.

### Pairformer

 The central transformer\-like stack in Protenix that processes pair embeddings \($z$\) and single embeddings \($s$\)\. It implements triangle\-based updates to maintain geometric consistency\.

 - **Triangle Multiplication:** Updates pair representations based on edges of a triangle \(Incoming/Outgoing\)\.
- **Triangle Attention:** Updates pair representations using axial attention over rows and columns\.

### Diffusion Module

 The generative component that iteratively refines atomic coordinates from noise\.

 - **Noising Schedule:** Defines how much Gaussian noise is added to the ground truth coordinates during training\.
- **Denoising \(x0 prediction\):** The model's attempt to predict the "clean" coordinates from a noisy input at time $t$\.

### Data Flow: From JSON to Features

 The following diagram illustrates how raw input is transformed into the internal tensors used by the model\.

 **Input Feature Pipeline Diagram**

  **Sources:**

 - `protenix/data/inference/json_to_feature.py` [37\-47](https://github.com/bytedance/Protenix/blob/c3bfc365/37-47)
- `protenix/model/modules/pairformer.py` [42\-102](https://github.com/bytedance/Protenix/blob/c3bfc365/42-102)

---

## 3\. Performance & Optimization Jargon

 Terms related to the efficient execution of Protenix on modern GPU hardware\.

 - **Fused Kernels:** Custom Triton\-based GPU operations that combine multiple steps \(e\.g\., Dropout \+ Residual Add\) into a single kernel to reduce memory bandwidth overhead\. - *Implementation:* `dropout_add_rowwise` in [fused\_ops\.py L190-L212](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/fused_ops.py#L190-L212)
- **Row\-wise Mask Sharing:** An optimization where the dropout mask is shared across the token dimension to improve consistency and reduce random number generation overhead\. - *Implementation:* `_dropout_add_rowwise_fwd_kernel` in [fused\_ops\.py L41-L58](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/fused_ops.py#L41-L58)
- **Inplace Operations:** Modifying tensors in\-place to save memory, specifically utilized in the Pairformer blocks during inference\. - *Pointer:* `inplace_safe` flag in [pairformer\.py L138-L169](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/model/modules/pairformer.py#L138-L169)
- **TF32 \(TensorFloat\-32\):** A math mode for NVIDIA Ampere\+ GPUs that provides speedups for matrix multiplications while maintaining sufficient precision for structure prediction\.

 **Sources:**

 - `protenix/model/modules/fused_ops.py` [15\-212](https://github.com/bytedance/Protenix/blob/c3bfc365/15-212)
- `protenix/model/modules/pairformer.py` [138\-169](https://github.com/bytedance/Protenix/blob/c3bfc365/138-169)
- `tests/test_fused_dropout_add.py` [49\-73](https://github.com/bytedance/Protenix/blob/c3bfc365/49-73)

---

## 4\. Training\-Free Guidance \(TFG\)

 A system for applying physical or geometric constraints during the diffusion sampling process without retraining the model\.

 - **Potential:** A differentiable energy function $E\(x\)$ representing a constraint \(e\.g\., "no clashes" or "stay near this pocket"\)\.
- **TFGEngine:** The runtime manager that calculates gradients of potentials and applies them to coordinates\.
- **Denoiser\-path Guidance \(rho\):** Applying guidance by backpropagating through the model's denoising function\.
- **Refinement \(mu\):** Directly shifting the predicted clean coordinates \($x\_0$\) based on the potential gradient\.

 **TFG Execution Flow**

  **Sources:**

 - `protenix/tfg/engine.py` [72\-133](https://github.com/bytedance/Protenix/blob/c3bfc365/72-133)
- `protenix/tfg/engine.py` [185\-206](https://github.com/bytedance/Protenix/blob/c3bfc365/185-206)

---

## 5\. MSA & Search Pipelines

 - **ColabFold\-compatible:** A mode for MSA generation that uses the MMseqs2 API, similar to the popular ColabFold tool\.
- **Template Search:** The process of finding known structures in the PDB that are homologous to the input sequence to use as spatial hints\.
- **RNA MSA:** Specialized alignment pipeline for RNA sequences using tools like `nhmmer`\.

 **Sources:**

 - `runner/batch_inference.py` [70\-111](https://github.com/bytedance/Protenix/blob/c3bfc365/70-111)
- `runner/msa_search.py` [1\-50](https://github.com/bytedance/Protenix/blob/c3bfc365/1-50)

---
*Source: [https://deepwiki.com/bytedance/Protenix/10-glossary](https://deepwiki.com/bytedance/Protenix/10-glossary) on DeepWiki*