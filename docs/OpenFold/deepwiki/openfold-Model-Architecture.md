---
title: "Model Architecture"
source: deepwiki.com
owner: aqlaboratory
repo: openfold
url: https://deepwiki.com/aqlaboratory/openfold/5-model-architecture
---
# Model Architecture

# Model Architecture

> **Relevant source files**
> - [openfold/config\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py)
> - [openfold/model/dropout\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/dropout.py)
> - [openfold/model/evoformer\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/evoformer.py)
> - [openfold/model/model\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py)
> - [openfold/model/template\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/template.py)
> - [openfold/model/triangular\_attention\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/triangular_attention.py)
> - [openfold/utils/tensor\_utils\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/utils/tensor_utils.py)

 The Model Architecture page describes the core neural network design and components that comprise the OpenFold implementation of AlphaFold 2 in PyTorch\. This document covers the overall model structure, key components, and their interactions during protein structure prediction\. For information about training the model, see [Training OpenFold](https://deepwiki.com/aqlaboratory/openfold/4-training-openfold); for details about running inference, see [Running Inference](https://deepwiki.com/aqlaboratory/openfold/3-running-inference)\.

## Overall Architecture Overview

 OpenFold's model architecture closely mirrors AlphaFold 2's design, implementing a deep learning system that predicts protein structures from sequence information, multiple sequence alignments \(MSAs\), and optional template information\.

```mermaid
flowchart TD

input["Input Features"]
msa["MSA Features"]
templ["Template Features"]
seq["Sequence Features"]
in_emb["InputEmbedder"]
rec_emb["RecyclingEmbedder"]
templ_emb["TemplateEmbedder"]
ex_msa_emb["ExtraMSAEmbedder"]
evo["EvoformerStack"]
ex_msa_stack["ExtraMSAStack"]
templ_stack["TemplatePairStack"]
struct["StructureModule"]
aux["AuxiliaryHeads"]
coords["3D Coordinates"]
m["MSA Representation (m)"]
z["Pair Representation (z)"]
m_out["Updated MSA"]
z_out["Updated Pair"]
s["Single Representation (s)"]
rec["Recycling Input"]

input --> in_emb
msa --> in_emb
seq --> in_emb
in_emb --> m
in_emb --> z
templ --> templ_emb
templ_stack --> z
ex_msa_stack --> z
m --> evo
z --> evo
evo --> m_out
evo --> z_out
evo --> s
m_out --> struct
z_out --> struct
s --> struct
m_out --> aux
z_out --> aux
s --> aux
coords --> rec
z_out --> rec
m_out --> rec
rec --> rec_emb
rec_emb --> m
rec_emb --> z

subgraph subGraph4 ["AlphaFold Model"]
    templ_emb --> templ_stack
    ex_msa_emb --> ex_msa_stack
    struct --> coords

subgraph Outputs ["Outputs"]
    aux
    coords
    coords --> aux
end

subgraph subGraph2 ["Processing Modules"]
    evo
    ex_msa_stack
    templ_stack
    struct
end

subgraph Embedders ["Embedders"]
    in_emb
    rec_emb
    templ_emb
    ex_msa_emb
end
end

subgraph subGraph0 ["Input Data"]
    input
    msa
    templ
    seq
end
```

 Sources: [model\.py L65-L492](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L65-L492) [config\.py L61-L260](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L61-L260)

## Model Components and Data Flow

 The OpenFold architecture consists of the following key components that process data through the system:

| Component | Purpose | Implementation |
| --- | --- | --- |
| InputEmbedder | Embeds sequence and MSA features into initial representations | model/embedders\.py |
| RecyclingEmbedder | Incorporates information from previous iterations | model/embedders\.py |
| TemplateEmbedder | Processes template information | model/template\.py |
| ExtraMSAEmbedder | Embeds additional MSA sequences | model/embedders\.py |
| EvoformerStack | Core processing module for MSA and pair representations | model/evoformer\.py |
| ExtraMSAStack | Processes extra MSA information | model/evoformer\.py |
| StructureModule | Converts embeddings to 3D coordinates | model/structure\_module\.py |
| AuxiliaryHeads | Produces additional outputs like confidence scores | model/heads\.py |

 The AlphaFold class \([model\.py L65-L591](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L65-L591)\) orchestrates these components, implementing the recycling mechanism that allows the model to refine its predictions over multiple iterations\.

## Code Structure and Class Relationships

 This diagram maps the model components to their corresponding classes in the codebase:

```mermaid
classDiagram
    class AlphaFold {
        forward(batch)
        iteration(feats, prevs)
        embed_templates(batch, z, pair_mask)
    }
    class InputEmbedder {
    }
    class RecyclingEmbedder {
    }
    class TemplateEmbedder {
    }
    class ExtraMSAEmbedder {
    }
    class ExtraMSAStack {
    }
    class EvoformerStack {
        c_m: int
        c_z: int
        no_blocks: int
        forward(m, z, msa_mask, pair_mask)
    }
    class StructureModule {
        forward(s, z, aatype, mask)
    }
    class AuxiliaryHeads {
    }
    class EvoformerBlock {
    }
    class MSARowAttentionWithPairBias {
    }
    class MSAColumnAttention {
    }
    class MSATransition {
    }
    class OuterProductMean {
    }
    class PairStack {
    }
    class TriangleMultiplicationOutgoing {
    }
    class TriangleMultiplicationIncoming {
    }
    class TriangleAttentionStartingNode {
    }
    class TriangleAttentionEndingNode {
    }
    class PairTransition {
    }
    class ExtraMSABlock {
    }
    class TemplatePairStack {
    }
    class TemplatePointwiseAttention {
    }
    class TemplatePairStackBlock {
    }
    class InvariantPointAttention {
    }
    class BackboneUpdate {
    }
    class AngleResnet {
    }
    class DistogramHead {
    }
    class PredictedLDDTHead {
    }
    class TMScoreHead {
    }
    class MaskedMSAHead {
    }
    AlphaFold --> InputEmbedder
    AlphaFold --> RecyclingEmbedder
    AlphaFold --> TemplateEmbedder
    AlphaFold --> ExtraMSAEmbedder
    AlphaFold --> ExtraMSAStack
    AlphaFold --> EvoformerStack
    AlphaFold --> StructureModule
    AlphaFold --> AuxiliaryHeads
    EvoformerStack --> EvoformerBlock
    EvoformerBlock --> MSARowAttentionWithPairBias
    EvoformerBlock --> MSAColumnAttention
    EvoformerBlock --> MSATransition
    EvoformerBlock --> OuterProductMean
    EvoformerBlock --> PairStack
    PairStack --> TriangleMultiplicationOutgoing
    PairStack --> TriangleMultiplicationIncoming
    PairStack --> TriangleAttentionStartingNode
    PairStack --> TriangleAttentionEndingNode
    PairStack --> PairTransition
    ExtraMSAStack --> ExtraMSABlock
    TemplateEmbedder --> TemplatePairStack
    TemplateEmbedder --> TemplatePointwiseAttention
    TemplatePairStack --> TemplatePairStackBlock
    StructureModule --> InvariantPointAttention
    StructureModule --> BackboneUpdate
    StructureModule --> AngleResnet
    AuxiliaryHeads --> DistogramHead
    AuxiliaryHeads --> PredictedLDDTHead
    AuxiliaryHeads --> TMScoreHead
    AuxiliaryHeads --> MaskedMSAHead
```

 Sources: [model\.py L65-L492](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L65-L492) [evoformer\.py L748-L1025](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/evoformer.py#L748-L1025)

## Model Initialization and Configuration

 The model is initialized with a comprehensive configuration object defined in [openfold/config\.py](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py) that controls all architectural aspects\. Different model presets are available, including:

 - Model variants \(1\-5\) with different architectural choices
- PTM variants with template modeling score prediction
- Multimer variants for multi\-chain protein complexes
- Sequence embedding variants for no\-MSA inference

 The `model_config` function in [config\.py L61-L260](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L61-L260) provides these preset configurations\.

```
model = AlphaFold(config)
```

## Input Processing and Embedding

 The model processes several types of input features:

 1. **Target sequence features**: One\-hot encoded sequence information
2. **MSA features**: Multiple sequence alignment information
3. **Template features**: Structural templates from similar proteins
4. **Extra MSA features**: Additional sequences for evolutionary information

 These inputs are embedded into initial representations by specialized embedders:

```mermaid
flowchart TD

input_standard["InputEmbedder"]
input_multimer["InputEmbedderMultimer"]
input_seqemb["PreembeddingEmbedder"]
seqemb_input["Sequence Embeddings"]
target_feat["Target Features"]
residue_index["Residue Index"]
msa_feat["MSA Features"]
multimer_inputs["Multimer Features"]
pair_repr["Pair Representation (z)"]
msa_repr["MSA Representation (m)"]
recycling["Recycling Embedder"]
prev_m["Previous MSA"]
prev_z["Previous Pair"]
prev_x["Previous Coords"]
pseudo_beta["Pseudo-Beta Calculation"]

seqemb_input --> input_seqemb
target_feat --> input_standard
residue_index --> input_standard
msa_feat --> input_standard
multimer_inputs --> input_multimer
input_standard --> pair_repr
input_standard --> msa_repr
input_multimer --> pair_repr
input_multimer --> msa_repr
input_seqemb --> pair_repr
input_seqemb --> msa_repr
recycling --> pair_repr
recycling --> msa_repr
prev_m --> recycling
prev_z --> recycling
prev_x --> pseudo_beta
pseudo_beta --> recycling

subgraph subGraph0 ["Input Embedders"]
    input_standard
    input_multimer
    input_seqemb
end
```

 Sources: [model\.py L87-L100](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L87-L100) [model\.py L232-L258](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L232-L258)

## The Evoformer Stack

 The Evoformer is the core processing module in OpenFold, responsible for iteratively refining the MSA and pair representations\. It consists of multiple EvoformerBlocks stacked together\.

```mermaid
flowchart TD

linear["Linear Projection"]
single["Single Representation (s)"]
tri_mul_out["Triangle Multiplication Outgoing"]
tri_mul_in["Triangle Multiplication Incoming"]
tri_att_start["Triangle Attention Starting Node"]
tri_att_end["Triangle Attention Ending Node"]
pair_trans["Pair Transition"]
msa_input["MSA Input (m)"]
msa_row["MSA Row Attention"]
pair_input["Pair Input (z)"]
msa_col["MSA Column Attention"]
msa_trans["MSA Transition"]
opm["Outer Product Mean"]
pair_update["Pair Update"]
pair_stack["Pair Stack"]
msa_output["MSA Output"]
pair_output["Pair Output"]
msa_in["Input MSA Representation"]
pair_in["Input Pair Representation"]
msa_out["Output MSA Representation"]
pair_out["Output Pair Representation"]
single_out["Single Representation"]

subgraph EvoformerStack ["EvoformerStack"]
    linear
    single
    linear --> single

subgraph subGraph1 ["EvoformerBlock 1..N"]
    msa_input
    msa_row
    pair_input
    msa_col
    msa_trans
    opm
    pair_update
    pair_stack
    msa_output
    pair_output
    msa_input --> msa_row
    pair_input --> msa_row
    msa_row --> msa_col
    msa_col --> msa_trans
    msa_trans --> opm
    opm --> pair_update
    pair_input --> pair_stack
    pair_update --> pair_stack
    msa_trans --> msa_output
    pair_stack --> pair_output

subgraph subGraph0 ["Pair Stack"]
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    pair_trans
    tri_mul_out --> tri_mul_in
    tri_mul_in --> tri_att_start
    tri_att_start --> tri_att_end
    tri_att_end --> pair_trans
end
end
end
```

 Sources: [evoformer\.py L747-L1025](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/evoformer.py#L747-L1025) [evoformer\.py L377-L552](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/evoformer.py#L377-L552)

 The EvoformerStack typically consists of 48 identical blocks, each performing several transformations on the MSA and pair representations:

 1. **MSA Row Attention with Pair Bias**: Attention between sequences, biased by pair information
2. **MSA Column Attention**: Attention between positions in the sequence
3. **MSA Transition**: Feed\-forward network applied to MSA
4. **Outer Product Mean**: Connects MSA and pair representations
5. **Pair Stack**: Series of operations on the pair representation - Triangle Multiplication: Outgoing and Incoming variants - Triangle Attention: Starting Node and Ending Node variants - Pair Transition: Feed\-forward network for pairs

## Template Processing

 Template information provides structural priors derived from similar proteins\. This information is processed through several specialized modules:

```mermaid
flowchart TD

embed_templates_offload["embed_templates_offload()"]
embed_templates_average["embed_templates_average()"]
template_feats["Template Features"]
template_pair_feat["build_template_pair_feat()"]
template_angle_feat["build_template_angle_feat()"]
template_pair_embedder["Template Pair Embedder"]
template_single_embedder["Template Single Embedder"]
template_pair_stack["Template Pair Stack"]
template_pointwise_att["Template Pointwise Attention"]
z["Pair Representation (z)"]
z_update["Updated Pair Representation"]
msa_update["Updated MSA Representation"]

template_feats --> template_pair_feat
template_feats --> template_angle_feat
template_pair_feat --> template_pair_embedder
template_angle_feat --> template_single_embedder
template_pair_embedder --> template_pair_stack
template_pair_stack --> template_pointwise_att
z --> template_pointwise_att
template_pointwise_att --> z_update
template_single_embedder --> msa_update

subgraph subGraph0 ["Memory Optimization Variants"]
    embed_templates_offload
    embed_templates_average
end
```

 Sources: [template\.py L136-L692](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/template.py#L136-L692) [model\.py L325-L342](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L325-L342)

 OpenFold implements two memory\-efficient template processing methods:

 1. `embed_templates_offload`: Offloads template tensors to CPU to save GPU memory
2. `embed_templates_average`: Processes templates in groups and averages them

## Structure Module

 The Structure Module transforms the embeddings from the Evoformer into 3D atomic coordinates, predicting the protein structure:

```mermaid
flowchart TD

ipa["Invariant Point Attention (IPA)"]
angle_resnet["Angle Resnet"]
backbone_update["Backbone Update"]
s["Single Representation (s)"]
struct_module["Structure Module"]
z["Pair Representation (z)"]
aatype["Amino Acid Types"]
final_atom_pos["Final Atom Positions"]
final_affine["Final Affine Tensor"]

s --> struct_module
z --> struct_module
aatype --> struct_module
struct_module --> final_atom_pos
struct_module --> final_affine

subgraph subGraph0 ["Structure Module"]
    ipa
    angle_resnet
    backbone_update
    ipa --> angle_resnet
    angle_resnet --> backbone_update
end
```

 Sources: [model\.py L442-L454](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L442-L454)

## Recycling Mechanism

 OpenFold implements an iterative refinement process called recycling\. In each recycling iteration:

 1. The model generates MSA, pair representations, and 3D coordinates
2. These outputs are used as inputs for the next iteration
3. The process continues for a fixed number of iterations or until convergence

```mermaid
sequenceDiagram
  participant Input Features
  participant AlphaFold Model
  participant Recycling Embedder
  participant Final Output

  Input Features->>AlphaFold Model: Initial features
  AlphaFold Model->>Recycling Embedder: MSA (m_1_prev)
  AlphaFold Model->>Recycling Embedder: Pair (z_prev)
  AlphaFold Model->>Recycling Embedder: Coords (x_prev)
  Recycling Embedder-->>AlphaFold Model: Updated embeddings
  loop [Recycling Iterations]
    Input Features->>AlphaFold Model: Features
    AlphaFold Model->>Recycling Embedder: Updated MSA (m_1_prev)
    AlphaFold Model->>Recycling Embedder: Updated Pair (z_prev)
    AlphaFold Model->>Recycling Embedder: Updated Coords (x_prev)
    Recycling Embedder-->>AlphaFold Model: Updated embeddings
  end
  AlphaFold Model->>Final Output: Final structure & confidence
```

 Sources: [model\.py L209-L473](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L209-L473) [model\.py L544-L586](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/model.py#L544-L586)

 The `recycle_early_stop_tolerance` parameter allows early stopping when consecutive iterations produce similar structures, improving efficiency\.

## Auxiliary Outputs

 In addition to the 3D structure, OpenFold produces several auxiliary outputs:

 1. **pLDDT**: Per\-residue confidence scores
2. **Distogram**: Distance distribution predictions between residues
3. **TM\-score**: Template modeling score prediction \(in PTM variants\)
4. **Masked MSA**: Used during training for MSA completion tasks

 These outputs are generated by specialized heads in the `AuxiliaryHeads` class\.

## Memory Optimization Techniques

 OpenFold incorporates several memory optimization techniques to enable processing of long sequences and multi\-chain complexes:

 1. **Activation checkpointing**: Reduces memory during backpropagation
2. **Chunk\-based processing**: Splits computations into manageable chunks
3. **Template offloading**: Moves template tensors to CPU when not needed
4. **Low\-memory attention**: Optional memory\-efficient attention implementations
5. **DeepSpeed integration**: For memory\-efficient attention kernels

 Sources: [evoformer\.py L876-L895](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/evoformer.py#L876-L895) [template\.py L472-L580](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/model/template.py#L472-L580) [config\.py L230-L236](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L230-L236)

## Model Variants and Configuration

 OpenFold supports various model configurations through the configuration system in [config\.py L61-L260](https://github.com/aqlaboratory/openfold/blob/56da08ec/openfold/config.py#L61-L260):

| Model Variant | Description |
| --- | --- |
| Model\_1 to Model\_5 | Different architectural settings matching AlphaFold2 presets |
| PTM variants | Include template modeling score prediction |
| Multimer variants | For multi\-chain protein complexes |
| Sequence embedding | For running without MSAs using ESM\-1b embeddings |

 These configurations control architectural choices like:

 - Number of Evoformer blocks
- Use of templates
- Size of extra MSA inputs
- Attention mechanism details

 The `model_config` function provides these preset configurations, which can be further customized\.

---
*Source: [https://deepwiki.com/aqlaboratory/openfold/5-model-architecture](https://deepwiki.com/aqlaboratory/openfold/5-model-architecture) on DeepWiki*