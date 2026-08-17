---
title: "ESM3 – Multimodal Protein Language Model"
source: deepwiki.com
owner: Biohub
repo: esm
url: https://deepwiki.com/Biohub/esm/2.1-esm3-multimodal-protein-language-model
---
# ESM3 – Multimodal Protein Language Model

# ESM3 – Multimodal Protein Language Model

> **Relevant source files**
> - [esm/models/esm3\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py)
> - [esm/models/function\_decoder\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/models/function_decoder.py)
> - [esm/pretrained\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/pretrained.py)
> - [esm/utils/structure/affine3d\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/affine3d.py)

 ESM3 is a multimodal generative protein language model capable of reasoning across sequence, structure, and functional annotations\. It is implemented as a transformer\-based architecture that integrates multiple data tracks into a unified latent space, enabling tasks such as structure prediction, inverse folding, and functional design\.

## ESM3 Architecture Overview

 The `ESM3` class in `esm/models/esm3.py` serves as the primary implementation of the model\. It inherits from `nn.Module` and implements the `ESM3InferenceClient` interface [esm3\.py L188](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L188-L188) The architecture consists of three main stages:

 1. **Input Encoding**: Projecting multimodal tracks \(sequence, structure tokens, etc\.\) into a shared embedding\.
2. **Transformer Backbone**: A stack of unified transformer blocks supporting both standard and geometric attention\.
3. **Output Heads**: Regressing logits for each track from the final transformer hidden states\.

### Data Flow and Code Entity Mapping

 The following diagram maps the high\-level multimodal data flow to the specific classes and modules within the ESM3 codebase\.

 **ESM3 Multimodal Pipeline**

  **Sources:** [esm3\.py L62-L185](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L62-L185) [transformer\_stack\.py L10-L116](https://github.com/Biohub/esm/blob/82ee3555/esm/layers/transformer_stack.py#L10-L116)

---

## Input Encoding \(EncodeInputs\)

 The `EncodeInputs` module handles the initial projection of eight distinct tracks into a single $d\_\{model\}$ vector per residue [esm3\.py L62-L70](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L62-L70)

| Track | Token Source | Encoding Mechanism |
| --- | --- | --- |
| Sequence | Amino Acid Tokens | nn\.Embedding\(64, d\_model\) esm/models/esm3\.py74 |
| Structure | VQ\-VAE Tokens | nn\.Embedding\(4096 \+ 5, d\_model\) esm/models/esm3\.py80 |
| SS8 | Secondary Structure | nn\.Embedding\(8 \+ 3, d\_model\) esm/models/esm3\.py83 |
| SASA | Solvent Accessibility | nn\.Embedding\(16 \+ 3, d\_model\) esm/models/esm3\.py84 |
| Function | InterPro LSH | nn\.ModuleList of 8 nn\.Embedding \(dim $d/8$\) esm/models/esm3\.py87\-89 |
| Residue | Annotations | nn\.EmbeddingBag\(1478, d\_model, mode="sum", padding\_idx=0\) esm/models/esm3\.py91 |
| pLDDT | Global/Per\-Res | RBF kernels \+ nn\.Linear projection esm/models/esm3\.py106\-112 |

 The forward pass sums these embeddings element\-wise: $$X\_\{input\} = E\_\{seq\} \+ E\_\{struct\} \+ E\_\{ss8\} \+ E\_\{sasa\} \+ E\_\{func\} \+ E\_\{res\} \+ E\_\{plddt\_avg\} \+ E\_\{plddt\_per\_res\}$$ [esm3\.py L139-L148](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L139-L148)

 **Sources:** [esm3\.py L62-L148](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L62-L148)

---

## Transformer Backbone and Geometric Conditioning

 The backbone is a `TransformerStack` composed of `UnifiedTransformerBlock` layers [esm3\.py L212](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L212-L212)

### UnifiedTransformerBlock

 Each block can perform two types of attention:

 1. **Standard Attention**: Multi\-head attention \(optionally using `FlashMultiHeadAttention`\)\.
2. **Geometric Attention**: Enabled via `use_geom_attn`\. It uses `GeometricReasoningOriginalImpl` to incorporate 3D spatial relationships\.

### Affine3D Conditioning

 Geometric attention is conditioned on `Affine3D` frames, which represent the local coordinate system \(orientation and position\) of each residue [esm3\.py L255-L265](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L255-L265) These frames are built from backbone coordinates during the forward pass using `build_affine3d_from_coordinates` [affine3d\.py L290-L300](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/affine3d.py#L290-L300)

 **Sources:** [esm3\.py L212](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L212-L212) [esm3\.py L255-L265](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L255-L265) [affine3d\.py L290-L300](https://github.com/Biohub/esm/blob/82ee3555/esm/utils/structure/affine3d.py#L290-L300)

---

## Output Heads and ESMOutput

 The `OutputHeads` module contains `RegressionHead` instances for each input track [esm3\.py L151-L160](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L151-L160)

 - **RegressionHead**: A standard block consisting of a Linear layer, GELU activation, LayerNorm, and a final Linear projection to the vocabulary size [regression\_head\.py L13-L20](https://github.com/Biohub/esm/blob/82ee3555/esm/layers/regression_head.py#L13-L20)
- **ESMOutput**: A dataclass that aggregates the resulting logits for sequence, structure, SS8, SASA, function, and residue tracks, along with the final embeddings and optional attention weights [esm3\.py L50-L59](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L50-L59)

 **Sources:** [esm3\.py L50-L59](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L50-L59) [esm3\.py L151-L185](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L151-L185) [regression\_head\.py L13-L20](https://github.com/Biohub/esm/blob/82ee3555/esm/layers/regression_head.py#L13-L20)

---

## Pretrained Factory Functions

 ESM3 models are typically instantiated using factory functions in `esm/pretrained.py` which handle architecture configuration and weight loading\.

 **Model Loading Logic**

  Key factory functions include:

 - `ESM3_sm_open_v0`: Loads the open\-source small ESM3 model \(1\.4B parameters, 48 layers, $d\_\{model\}=1536$\) [pretrained\.py L108-L125](https://github.com/Biohub/esm/blob/82ee3555/esm/pretrained.py#L108-L125)
- `ESM3_structure_encoder_v0` / `ESM3_structure_decoder_v0`: Loads the VQ\-VAE components for structural tokenization [pretrained\.py L28-L51](https://github.com/Biohub/esm/blob/82ee3555/esm/pretrained.py#L28-L51)
- `ESM3_function_decoder_v0`: Loads the decoder for InterPro and keyword prediction [pretrained\.py L54-L63](https://github.com/Biohub/esm/blob/82ee3555/esm/pretrained.py#L54-L63)

 **Sources:** [pretrained\.py L28-L136](https://github.com/Biohub/esm/blob/82ee3555/esm/pretrained.py#L28-L136)

---

## Forward Pass Implementation

 The `ESM3.forward()` method orchestrates the multimodal data flow [esm3\.py L238-L306](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L238-L306):

 1. **Coordinate Processing**: If raw coordinates are provided, it builds `Affine3D` frames and calculates per\-residue pLDDT [esm3\.py L255-L265](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L255-L265)
2. **Encoding**: Calls `self.encoder` to generate the initial latent representation [esm3\.py L276-L286](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L276-L286)
3. **Transformer Stack**: Passes the embeddings through `self.transformer`, applying geometric attention if structural frames are present [esm3\.py L288-L295](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L288-L295)
4. **Heads**: Passes the transformer output to `self.heads` to generate `ESMOutput` [esm3\.py L297-L306](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L297-L306)

 **Sources:** [esm3\.py L238-L306](https://github.com/Biohub/esm/blob/82ee3555/esm/models/esm3.py#L238-L306)

---
*Source: [https://deepwiki.com/Biohub/esm/2.1-esm3-multimodal-protein-language-model](https://deepwiki.com/Biohub/esm/2.1-esm3-multimodal-protein-language-model) on DeepWiki*