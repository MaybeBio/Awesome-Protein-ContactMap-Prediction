---
title: "Sequence & Secondary Structure Tokenizers"
source: deepwiki.com
owner: Biohub
repo: esm
url: https://deepwiki.com/Biohub/esm/5.1-sequence-and-secondary-structure-tokenizers
---
# Sequence & Secondary Structure Tokenizers

# Sequence & Secondary Structure Tokenizers

> **Relevant source files**
> - [esm/tokenization/residue\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/residue_tokenizer.py)
> - [esm/tokenization/sasa\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py)
> - [esm/tokenization/sequence\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py)
> - [esm/tokenization/ss\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py)
> - [esm/tokenization/structure\_tokenizer\.py](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py)

 This page documents the tokenization system for sequence and structural tracks in ESM\. The system is built around a common protocol, `EsmTokenizerBase`, which ensures consistency across different data modalities including amino acid sequences, secondary structure classifications, and discretized surface accessibility values\.

## Tokenization Architecture

 The ESM tokenization system uses a multimodal approach where different tracks \(sequence, structure, SASA, etc\.\) are tokenized independently before being processed by the transformer\.

### EsmTokenizerBase Protocol

 All tokenizers in the ESM ecosystem implement the `EsmTokenizerBase` protocol\. This ensures that every tokenizer provides a consistent interface for special tokens and basic operations like encoding and decoding\.

 **Interface Overview:**

 - **Special Tokens**: Every tokenizer must provide `mask`, `bos`, `eos`, `pad`, and `chain_break` tokens and their corresponding integer IDs [tokenizer\_base\.py L5-L15](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/tokenizer_base.py#L5-L15)
- **Methods**: Implementations must provide `encode()` and `decode()` methods [tokenizer\_base\.py L17-L19](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/tokenizer_base.py#L17-L19)
- **Properties**: `all_token_ids` and `special_token_ids` are required for sampling and masking logic [tokenizer\_base\.py L21-L25](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/tokenizer_base.py#L21-L25)

### Sequence & Structure Tokenization Flow

 The following diagram illustrates how raw protein data is transformed into tokens by the various tokenizer classes\.

 **Data to Token Entity Mapping**

  Sources: [sequence\_tokenizer\.py L10-L15](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L10-L15) [ss\_tokenizer\.py L10-L13](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L10-L13) [sasa\_tokenizer\.py L9-L12](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L9-L12)

---

## EsmSequenceTokenizer

 The `EsmSequenceTokenizer` is the primary tokenizer for amino acid sequences\. It inherits from `PreTrainedTokenizerFast` \(HuggingFace\) and uses a character\-level Byte Pair Encoding \(BPE\) model with no merges [sequence\_tokenizer\.py L10-L31](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L10-L31)

### Configuration and Vocab

 - **Vocabulary**: Based on `C.SEQUENCE_VOCAB` [sequence\_tokenizer\.py L27-L28](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L27-L28)
- **Special Tokens**: Includes `<unk>`, `<cls>`, `<pad>`, `<mask>`, `<eos>`, and the chain break token `|` [sequence\_tokenizer\.py L18-L24](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L18-L24)
- **Post\-Processing**: Automatically wraps single sequences in `<cls>` and `<eos>` tokens using `TemplateProcessing` [sequence\_tokenizer\.py L48-L54](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L48-L54)

### Chain Breaks

 The tokenizer handles multi\-chain proteins using the `|` character\. This is mapped to a specific `chain_break_token_id` to inform the model of discontinuities in the polypeptide backbone [sequence\_tokenizer\.py L24-L41](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L24-L41)

 Sources: [sequence\_tokenizer\.py L10-L64](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L10-L64)

---

## SecondaryStructureTokenizer

 The `SecondaryStructureTokenizer` handles the discretization of protein secondary structure into either 8\-class \(SS8\) or 3\-class \(SS3\) representations [ss\_tokenizer\.py L13-L15](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L13-L15)

### Supported Vocabularies

| Kind | Vocabulary Source | Tokens |
| --- | --- | --- |
| SS8 | C\.SSE\_8CLASS\_VOCAB | G, H, I, T, E, B, S, C esm/tokenization/ss\_tokenizer\.py26 |
| SS3 | C\.SSE\_3CLASS\_VOCAB | H, E, C esm/tokenization/ss\_tokenizer\.py28 |

### Implementation Details

 - **Special Tokens**: Uses `<pad>`, `<motif>`, and `<unk>`\. Notably, for secondary structure, `<pad>` is used for BOS, EOS, and Chainbreak roles [ss\_tokenizer\.py L80-L117](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L80-L117)
- **Encoding**: Maps characters directly to indices in the concatenated list of special and non\-special tokens [ss\_tokenizer\.py L33-L67](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L33-L67)

 Sources: [ss\_tokenizer\.py L10-L126](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L10-L126)

---

## SASADiscretizingTokenizer

 The `SASADiscretizingTokenizer` converts continuous Solvent Accessible Surface Area \(SASA\) values into discrete bins\.

### Discretization Logic

 - **Bins**: Uses 15 discretization boundaries defined in `C.SASA_DISCRETIZATION_BOUNDARIES` [sasa\_tokenizer\.py L12](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L12-L12)
- **Bucketing**: Uses `torch.bucketize` to map float values into discrete ranges [sasa\_tokenizer\.py L79-L80](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L79-L80)
- **Midpoints**: For decoding, the tokenizer maintains a `midpoints_tensor` to map discrete tokens back to a representative float value \(the center of the bin\) [sasa\_tokenizer\.py L34-L42](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L34-L42)

### Vocabulary Structure

 The vocabulary consists of special tokens followed by range tokens formatted as `<low-high>` \(e\.g\., `<0-2.5>`\) [sasa\_tokenizer\.py L26-L31](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L26-L31)

 Sources: [sasa\_tokenizer\.py L9-L106](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L9-L106)

---

## StructureTokenizer \(VQ\-VAE Wrapper\)

 The `StructureTokenizer` is a convenience class designed to provide access to special token IDs used by the `StructureTokenEncoder` and `StructureTokenDecoder` \(the VQ\-VAE system\) [structure\_tokenizer\.py L5-L7](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py#L5-L7)

### VQ\-VAE Special Tokens

 Because structure tokens represent latent codes from a VQ\-VAE codebook, they do not have string representations\. The tokenizer defines special IDs starting after the codebook size [structure\_tokenizer\.py L9-L16](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py#L9-L16)

| Token Name | ID Calculation |
| --- | --- |
| MASK | codebook\_size |
| EOS | codebook\_size \+ 1 |
| BOS | codebook\_size \+ 2 |
| PAD | codebook\_size \+ 3 |
| CHAINBREAK | codebook\_size \+ 4 |

 **Note**: Calling `encode()` or `decode()` on this class will raise a `NotImplementedError`, as structure encoding requires 3D coordinates and the VQ\-VAE model [structure\_tokenizer\.py L71-L83](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py#L71-L83)

 Sources: [structure\_tokenizer\.py L5-L83](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py#L5-L83)

---

## Class Relationship Diagram

 The following diagram shows how the tokenizer classes relate to the base protocol and the underlying utilities\.

 **Tokenizer Class Hierarchy**

  Sources: [tokenizer\_base\.py L4-L26](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/tokenizer_base.py#L4-L26) [sequence\_tokenizer\.py L10](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sequence_tokenizer.py#L10-L10) [ss\_tokenizer\.py L10](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/ss_tokenizer.py#L10-L10) [sasa\_tokenizer\.py L9](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/sasa_tokenizer.py#L9-L9) [structure\_tokenizer\.py L5](https://github.com/Biohub/esm/blob/82ee3555/esm/tokenization/structure_tokenizer.py#L5-L5)

---
*Source: [https://deepwiki.com/Biohub/esm/5.1-sequence-and-secondary-structure-tokenizers](https://deepwiki.com/Biohub/esm/5.1-sequence-and-secondary-structure-tokenizers) on DeepWiki*