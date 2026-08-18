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
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/msa/mmseqs2\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/main\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py)

## Purpose and Scope

 This document explains how Multiple Sequence Alignments \(MSAs\) are generated and provided to the Boltz prediction pipeline\. MSAs are critical for protein structure prediction as they provide evolutionary context that improves accuracy\.

 This page covers:

 - Automatic MSA generation using the MMseqs2 server API
- Authentication methods for MSA server access
- Custom MSA file formats \(A3M and CSV\)
- The workflow from input sequences to processed MSA data

 For information about how MSAs are parsed and processed into model features, see [MSA Processing](https://deepwiki.com/jwohlwend/boltz/4.4-msa-processing)\. For details on how MSA features are used in the model, see [Feature Generation](https://deepwiki.com/jwohlwend/boltz/4.3-feature-generation)\.

---

## Overview

 Boltz supports two approaches for providing MSAs:

 1. **Automatic Generation**: When the `--use_msa_server` flag is set, Boltz automatically generates MSAs by querying the MMseqs2 server API with protein sequences\.
2. **Pre\-computed MSAs**: Users can provide custom MSA files in A3M or CSV format via the input YAML/FASTA specification\.

 MSA generation only applies to protein sequences\. DNA, RNA, and ligand entities do not use MSAs and their `msa_id` fields are set to `-1` during processing\.

 **Sources:** [main\.py L925-L928](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L925-L928) [main\.py L566-L579](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L566-L579)

---

## Automatic MSA Generation via MMseqs2

### Workflow

 When `--use_msa_server` is enabled, Boltz uses the MMseqs2 server to generate MSAs for all protein chains that don't have pre\-specified MSA files\. The process involves:

  **Figure 1: MSA Generation Workflow**

 The `compute_msa()` function orchestrates this process:

 1. **Paired MSA Generation** \(multi\-chain complexes only\): Generates paired MSAs where sequences across chains are co\-evolved and aligned\. This uses the pairing strategy specified by `--msa_pairing_strategy`\.
2. **Unpaired MSA Generation**: Generates standard MSAs for each chain independently using environmental databases\.
3. **Combination**: Merges paired and unpaired MSAs, prioritizing paired sequences up to `const.max_paired_seqs` \(default: as defined in constants\), then filling remaining slots up to `const.max_msa_seqs` with unpaired sequences\.

 **Sources:** [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L523) [main\.py L469-L495](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L469-L495)

---

### MMseqs2 Server API Interaction

 The `run_mmseqs2()` function in `mmseqs2.py` handles all communication with the MMseqs2 server:

  **Figure 2: MMseqs2 API Interaction Sequence**

 **Key Implementation Details:**

 - **Endpoints**: - Paired mode: `ticket/pair` - Unpaired mode: `ticket/msa`
- **Modes**: Constructed from `use_env`, `use_filter`, and `pairing_strategy`: - `"env"`: Environmental databases with filtering - `"pairgreedy"`, `"paircomplete"`: Pairing strategies - `"env-nofilter"`: No filtering applied
- **Retry Logic**: All API calls include retry logic with exponential backoff \(up to 5 attempts\)
- **Progress Tracking**: Uses `tqdm` progress bars with estimated completion times

 **Sources:** [mmseqs2\.py L21-L286](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L21-L286) [mmseqs2\.py L65-L104](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L65-L104)

---

### Pairing Strategies

 The `--msa_pairing_strategy` option controls how sequences are paired across multiple chains:

| Strategy | Behavior | Use Case |
| --- | --- | --- |
| greedy | Fast pairing using greedy matching algorithm | Default, recommended for most cases |
| complete | Exhaustive pairing \(previous default\) | More thorough but slower |

 Pairing is only performed when multiple protein chains exist in the complex\. Single\-chain predictions skip the paired MSA generation step\.

 **Sources:** [mmseqs2\.py L169-L177](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L169-L177) [prediction\.md?plain=1 L936-L943](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L936-L943)

---

## Authentication

 The MSA server may require authentication\. Boltz supports two mutually exclusive authentication methods:

### Basic Authentication

 Uses username and password credentials:

 **CLI Options:**

  **Environment Variables:**

  Environment variables are used if CLI options are not provided\. The password should preferably be set via environment variable for security\.

 **Implementation:** Uses `requests.auth.HTTPBasicAuth` to construct the `Authorization` header\.

 **Sources:** [main\.py L1115-L1129](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L1115-L1129) [mmseqs2\.py L49-L52](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L49-L52)

---

### API Key Authentication

 Uses custom header\-based authentication:

 **CLI Options:**

  **Environment Variable:**

  The `api_key_header` specifies the HTTP header name \(defaults to `X-API-Key`\), and `api_key_value` provides the key itself\. The environment variable is checked if the CLI option is not provided\.

 **Implementation:** Constructs a dictionary of headers passed to all requests:

  **Sources:** [main\.py L454-L463](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L454-L463) [mmseqs2\.py L53-L55](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L53-L55)

---

### Authentication Validation

 **Mutual Exclusivity:** The system validates that only one authentication method is used\. If both basic auth and API key auth are provided, an error is raised:

  This validation occurs in both `process_inputs()` and `run_mmseqs2()`\.

 **Sources:** [main\.py L714-L722](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L714-L722) [mmseqs2\.py L35-L42](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L35-L42)

---

## Custom MSA File Formats

 Users can provide pre\-computed MSAs instead of using automatic generation\. Boltz supports two formats:

### A3M Format

 The A3M format is a standard MSA format where:

 - Header lines start with `>`
- Sequence lines follow headers
- Lowercase letters represent insertions
- Dashes \(`-`\) represent deletions

 **Specification in YAML:**

  **Processing:** A3M files are parsed by `parse_a3m()` which extracts sequences, deletions, and optional taxonomy information\.

 **Sources:** [main\.py L614-L620](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L614-L620) [prediction\.md?plain=1 L74-L77](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L74-L77)

---

### CSV Format with Pairing Keys

 For multi\-chain complexes where sequences should be paired, use the CSV format:

 **Structure:**

  **Fields:**

 - `key`: Integer identifier indicating pairing\. Sequences with the same key across different CSV files are mutually aligned\.
- `sequence`: The amino acid sequence \(may include dashes for gaps\)

 **Key Values:**

 - Non\-negative integers \(`0`, `1`, `2`, \.\.\.\): Paired sequences
- `-1`: Unpaired sequences

 **Example for Two\-Chain Complex:**

 Chain A MSA \(`chain_a.csv`\):

  Chain B MSA \(`chain_b.csv`\):

  Sequences with `key=0` and `key=1` are co\-evolved and will be paired during MSA processing\.

 **Sources:** [prediction\.md?plain=1 L74-L77](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L74-L77) [main\.py L621-L625](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L621-L625)

---

## MSA Generation Workflow in Code

  **Figure 3: MSA Generation Integration in Input Processing**

### Key Functions

| Function | Location | Purpose |
| --- | --- | --- |
| process\_input\(\) | src/boltz/main\.py525\-662 | Orchestrates input parsing and MSA generation for a single input file |
| compute\_msa\(\) | src/boltz/main\.py415\-523 | Calls MMseqs2 server and combines paired/unpaired results |
| run\_mmseqs2\(\) | src/boltz/data/msa/mmseqs2\.py21\-286 | Interfaces with MMseqs2 server API |
| parse\_a3m\(\) | Referenced in src/boltz/main\.py615\-620 | Parses A3M format MSAs |
| parse\_csv\(\) | Referenced in src/boltz/main\.py622 | Parses CSV format MSAs with pairing keys |

 **Sources:** [main\.py L525-L662](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L525-L662) [main\.py L415-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L415-L523)

---

## Configuration Options

 The following CLI options control MSA generation:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| \-\-use\_msa\_server | Flag | False | Enable automatic MSA generation via MMseqs2 server |
| \-\-msa\_server\_url | String | https://api\.colabfold\.com | MSA server endpoint URL |
| \-\-msa\_pairing\_strategy | String | greedy | Pairing strategy: greedy or complete |
| \-\-msa\_server\_username | String | None | Username for basic authentication |
| \-\-msa\_server\_password | String | None | Password for basic authentication |
| \-\-api\_key\_header | String | None | Header name for API key auth \(default: X\-API\-Key\) |
| \-\-api\_key\_value | String | None | API key value for authentication |
| \-\-max\_msa\_seqs | Integer | 8192 | Maximum number of MSA sequences to retain |
| \-\-preprocessing\_threads | Integer | CPU count | Number of parallel threads for processing inputs |

 **Environment Variables:**

| Variable | Purpose |
| --- | --- |
| BOLTZ\_MSA\_USERNAME | MSA server username \(basic auth\) |
| BOLTZ\_MSA\_PASSWORD | MSA server password \(basic auth\) |
| MSA\_API\_KEY\_VALUE | API key value for header\-based auth |

 **Sources:** [main\.py L925-L1020](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L925-L1020) [prediction\.md?plain=1 L142-L173](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L142-L173)

---

## MSA File Output Structure

 Generated MSA files are stored in the output directory with the following structure:

```python
out_dir/
├── msa/                                    # Raw MSA files from server
│   ├── {target_id}_{entity_id}.csv        # Generated MSAs in CSV format
│   └── {target_id}_paired_tmp/            # Temporary paired MSA data
│       └── pair.a3m
│   └── {target_id}_unpaired_tmp/          # Temporary unpaired MSA data
│       └── uniref.a3m
│       └── bfd.mgnify30.metaeuk30.smag30.a3m
└── processed/
    └── msa/                                # Processed MSAs ready for model
        └── {target_id}_{msa_idx}.npz      # Binary MSA data
```

 **CSV Format Output:**

 Generated MSA files use the CSV format internally:

  Where:

 - Paired sequences have non\-negative keys \(`0`, `1`, `2`, \.\.\.\)
- Unpaired sequences have key `-1`
- Sequences are ordered: paired first \(up to `const.max_paired_seqs`\), then unpaired

 **Sources:** [main\.py L496-L523](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L496-L523) [main\.py L745-L762](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L745-L762)

---

## Special Cases

### Single Sequence Mode

 To force single\-sequence prediction \(not recommended\), specify `msa: empty` in the YAML:

  This bypasses MSA generation and processing entirely\.

 **Sources:** [prediction\.md?plain=1 L74-L77](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1#L74-L77)

---

### Non\-Protein Entities

 DNA, RNA, and ligand entities do not use MSAs\. During processing, their `msa_id` is set to `-1`:

  **Sources:** [main\.py L576-L578](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L576-L578)

---

## Error Handling

### Common Error Scenarios

 1. **Missing MSAs without server flag:**   ```yaml RuntimeError: Missing MSA's in input and --use_msa_server flag not set. ```  **Solution:** Either provide MSA files or enable `--use_msa_server`\.
2. **MSA file not found:**   ```yaml FileNotFoundError: MSA file {msa_path} not found. ```  **Solution:** Verify the MSA file path is correct and accessible\.
3. **Unsupported MSA format:**   ```yaml RuntimeError: MSA file {msa_path} not supported, only a3m or csv. ```  **Solution:** Convert MSA to A3M or CSV format\.
4. **Server authentication conflict:**   ```yaml ValueError: Cannot use both basic authentication and API key authentication. ```  **Solution:** Use only one authentication method\.
5. **MMseqs2 server errors:**  - `RATELIMIT`: Too many requests, automatically retries with backoff - `MAINTENANCE`: Server under maintenance - `ERROR`: Invalid input sequence or persistent server issues

 **Sources:** [main\.py L581-L625](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L581-L625) [mmseqs2\.py L202-L251](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/msa/mmseqs2.py#L202-L251)

---

## Implementation Details

### MSA ID Mapping

 During processing, MSA files are renamed and indexed:

 1. **Initial State:** `msa_id` points to the original file path \(e\.g\., `./msa/seq1.a3m`\)
2. **Processing:** MSA is parsed and dumped to `processed/msa/{target_id}_{msa_idx}.npz`
3. **Update:** `chain.msa_id` is updated to the new identifier string \(e\.g\., `"target_123_0"`\)

 This mapping is tracked in `msa_id_map` dictionary during processing\.

 **Sources:** [main\.py L600-L632](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L600-L632)

---

### Multiprocessing

 When `--preprocessing_threads > 1`, MSA generation is parallelized:

  Each input file is processed independently in parallel, including its MSA generation if needed\.

 **Sources:** [main\.py L794-L803](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/main.py#L794-L803)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation](https://deepwiki.com/jwohlwend/boltz/2.3-msa-generation) on DeepWiki*