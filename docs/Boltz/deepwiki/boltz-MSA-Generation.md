---
title: "MSA Generation"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation
---
# MSA Generation

# MSA Generation

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/data/parse/a3m\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py)

## Purpose and Scope

 This document explains how Multiple Sequence Alignments \(MSAs\) are generated and provided to the Boltz prediction pipeline\. MSAs are critical for protein structure prediction as they provide evolutionary context that improves accuracy\.

 This page covers:

 - Automatic MSA generation using the MMseqs2 server API\.
- Authentication methods for MSA server access \(Basic Auth vs\. API Keys\)\.
- Custom MSA file formats \(A3M and CSV\)\.
- The workflow from input sequences to processed MSA data\.

 For information about how MSAs are parsed and processed into model features, see [MSA Processing](https://github.com/jwohlwend/boltz/blob/b1ebfc46/MSA Processing) For details on how MSA features are used in the model, see [Feature Generation](https://github.com/jwohlwend/boltz/blob/b1ebfc46/Feature Generation)

---

## Overview

 Boltz supports two approaches for providing MSAs:

 1. **Automatic Generation**: When the `--use_msa_server` flag is set, Boltz automatically generates MSAs by querying the MMseqs2 server API with protein sequences [prediction\.md?plain=1 L7-L9](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L7-L9)
2. **Pre\-computed MSAs**: Users can provide custom MSA files in A3M or CSV format via the input YAML specification [prediction\.md?plain=1 L73-L78](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L73-L78)

 MSA generation only applies to protein sequences\. DNA, RNA, and ligand entities do not use MSAs and their `msa_id` fields are set to `-1` during processing [main\.py L576-L578](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L576-L578)

 **Sources:** [main\.py L576-L578](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L576-L578) [prediction\.md?plain=1 L7-L9](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L7-L9) [prediction\.md?plain=1 L73-L78](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L73-L78)

---

## Automatic MSA Generation via MMseqs2

### Workflow

 When `--use_msa_server` is enabled, Boltz uses the MMseqs2 server to generate MSAs for all protein chains that don't have pre\-specified MSA files\. The process involves:

  **Figure 1: MSA Generation Workflow**

 The `compute_msa` function in `src/boltz/main.py` orchestrates this process [main\.py L415-L416](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L415-L416):

 1. **Paired MSA Generation** \(multi\-chain complexes only\): Generates paired MSAs where sequences across chains are co\-evolved and aligned\. This uses the pairing strategy specified by `pairing_strategy` [main\.py L469-L482](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L469-L482)
2. **Unpaired MSA Generation**: Generates standard MSAs for each chain independently using environmental databases [main\.py L484-L495](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L484-L495)
3. **Combination**: Merges paired and unpaired MSAs, prioritizing paired sequences up to `const.max_paired_seqs`, then filling remaining slots up to `const.max_msa_seqs` with unpaired sequences [main\.py L503-L518](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L503-L518)

 **Sources:** [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L415-L523) [main\.py L469-L518](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L469-L518)

---

### MMseqs2 Server API Interaction

 The `run_mmseqs2` function in `src/boltz/data/msa/mmseqs2.py` handles all communication with the MMseqs2 server [mmseqs2\.py L21-L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L32):

  **Figure 2: MMseqs2 API Interaction Sequence**

 **Key Implementation Details:**

 - **Endpoints**: - Paired mode: `ticket/pair` [mmseqs2\.py L33](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L33-L33) - Unpaired mode: `ticket/msa` [mmseqs2\.py L33](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L33-L33)
- **Retry Logic**: The `submit`, `status`, and `download` internal functions include retry logic \(up to 5 attempts\) for network failures [mmseqs2\.py L88-L94](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L88-L94) [mmseqs2\.py L117-L124](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L117-L124) [mmseqs2\.py L146-L153](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L146-L153)
- **Progress Tracking**: Uses `tqdm` with a specific `TQDM_BAR_FORMAT` to show elapsed and remaining time [mmseqs2\.py L16-L19](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L16-L19) [mmseqs2\.py L196-L198](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L196-L198)

 **Sources:** [mmseqs2\.py L21-L286](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L286) [mmseqs2\.py L33-L153](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L33-L153)

---

### Pairing Strategies

 The `pairing_strategy` argument controls how sequences are paired across multiple chains [mmseqs2\.py L27](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L27-L27):

| Strategy | Behavior | Use Case |
| --- | --- | --- |
| greedy | Fast pairing using greedy matching algorithm | Default, recommended for most cases src/boltz/data/msa/mmseqs2\.py172\-173 |
| complete | Exhaustive pairing | More thorough but slower src/boltz/data/msa/mmseqs2\.py174\-175 |

 **Sources:** [mmseqs2\.py L169-L178](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L169-L178)

---

## Authentication

 The MMseqs2 server may require authentication\. Boltz supports two mutually exclusive methods [mmseqs2\.py L35-L42](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L35-L42):

### Basic Authentication

 Uses `HTTPBasicAuth` with a username and password\.

 - **Implementation**: `HTTPBasicAuth(msa_server_username, msa_server_password)` [mmseqs2\.py L50-L51](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L50-L51)

### Header\-based \(API Key\) Authentication

 Uses custom headers \(e\.g\., for API keys\)\.

 - **Implementation**: The `auth_headers` dictionary is updated into the request headers [mmseqs2\.py L53-L54](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L53-L54)

 **Validation Logic**: Boltz raises a `ValueError` if both basic auth and header auth are provided simultaneously [mmseqs2\.py L38-L42](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L38-L42)

 **Sources:** [mmseqs2\.py L35-L55](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L35-L55)

---

## Custom MSA File Formats

### A3M Format

 Standard format where lowercase letters represent insertions and dashes represent deletions\.

 - **Parsing**: Handled by `_parse_a3m` in `src/boltz/data/parse/a3m.py` [a3m\.py L11-L15](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L11-L15)
- **Data Structures**: Converts sequences into `MSAResidue`, `MSADeletion`, and `MSASequence` numpy arrays [a3m\.py L96-L100](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L96-L100)

### CSV Format \(Paired MSAs\)

 For manual pairing, users can provide a CSV with `key` and `sequence` columns [prediction\.md?plain=1 L76](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L76-L76)

 - **Key Logic**: Sequences with the same `key` \(integer\) across different chains are mutually aligned\.
- **Key \-1**: Indicates unpaired sequences [prediction\.md?plain=1 L76](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L76-L76)

 **Sources:** [a3m\.py L11-L101](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L11-L101) [prediction\.md?plain=1 L73-L78](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L73-L78)

---

## MSA Generation Workflow in Code

  **Figure 3: Integration of MSA Generation in the Prediction Pipeline**

### Key Functions

| Function | Location | Purpose |
| --- | --- | --- |
| run\_mmseqs2 | src/boltz/data/msa/mmseqs2\.py | Low\-level API client for MMseqs2 server src/boltz/data/msa/mmseqs2\.py21 |
| compute\_msa | src/boltz/main\.py | High\-level logic for paired vs unpaired generation src/boltz/main\.py415 |
| parse\_a3m | src/boltz/data/parse/a3m\.py | Parser for A3M files \(handles \.gz and plain text\) src/boltz/data/parse/a3m\.py104 |
| process\_input | src/boltz/main\.py | Main entry point for preparing target data src/boltz/main\.py525 |

 **Sources:** [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/main.py#L415-L523) [mmseqs2\.py L21-L32](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/msa/mmseqs2.py#L21-L32) [a3m\.py L104-L134](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/a3m.py#L104-L134)

---

## Special Cases

### Single Sequence Mode

 To bypass MSA usage for a protein chain, set `msa: empty` in the input YAML [prediction\.md?plain=1 L77](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L77-L77) This forces the model to rely solely on the primary sequence, which typically reduces accuracy\.

### PDB Templates and MSAs

 While MSAs provide evolutionary info, structural templates are handled separately via `parse_pdb` [pdb\.py L7-L13](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py#L7-L13) `parse_pdb` converts PDB files to mmCIF internally before extraction [pdb\.py L14-L31](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py#L14-L31)

 **Sources:** [prediction\.md?plain=1 L77](https://github.com/jwohlwend/boltz/blob/b1ebfc46/docs/prediction.md?plain=1#L77-L77) [pdb\.py L7-L31](https://github.com/jwohlwend/boltz/blob/b1ebfc46/src/boltz/data/parse/pdb.py#L7-L31)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation](https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation) on DeepWiki*