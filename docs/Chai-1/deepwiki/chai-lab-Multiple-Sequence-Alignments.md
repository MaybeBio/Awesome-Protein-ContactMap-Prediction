---
title: "Multiple Sequence Alignments"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments
---
# Multiple Sequence Alignments

# Multiple Sequence Alignments

> **Relevant source files**
> - [chai\_lab/data/dataset/msas/colabfold\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py)
> - [chai\_lab/data/dataset/msas/load\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/load.py)
> - [chai\_lab/data/dataset/msas/msa\_context\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/msa_context.py)
> - [chai\_lab/data/dataset/msas/preprocess\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py)
> - [chai\_lab/data/features/generators/msa\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/msa.py)
> - [chai\_lab/data/parsing/fasta\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/fasta.py)
> - [chai\_lab/data/parsing/msas/a3m\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py)
> - [chai\_lab/data/parsing/msas/aligned\_pqt\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py)
> - [chai\_lab/data/parsing/msas/data\_source\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/data_source.py)
> - [chai\_lab/data/parsing/msas/species\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/species.py)
> - [chai\_lab/data/parsing/restraints\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/restraints.py)
> - [chai\_lab/model/utils\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/model/utils.py)
> - [examples/msas/predict\_with\_msas\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/examples/msas/predict_with_msas.py)
> - [scripts/stage\_colabfold\_outputs\_for\_chai\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/scripts/stage_colabfold_outputs_for_chai.py)
> - [tests/test\_colabfold\_msas\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_colabfold_msas.py)

 Multiple Sequence Alignments \(MSAs\) provide evolutionary information that significantly improves protein structure prediction in Chai\-1\. This page documents how MSAs are formatted, generated, processed, and integrated into the prediction pipeline\.

## MSA Data Format

 Chai\-1 uses a custom `.aligned.pqt` file format\. This is a Parquet dataframe that extends standard alignment information with metadata for pairing and source tracking\.

### AlignedParquetModel Schema

 The schema is enforced using `AlignedParquetModel` defined in [aligned\_pqt\.py L37-L44](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L37-L44)

| Column | Type | Description |
| --- | --- | --- |
| sequence | str | Aligned sequence in a3m format \(including gaps \- and deletions\)\. |
| source\_database | str | Database source \(e\.g\., uniref90, mgnify, query\)\. |
| pairing\_key | str | Key for pairing alignments across chains \(often species names\)\. |
| comment | str | Metadata or header information from the original hit\. |

 The first row of any `.aligned.pqt` file must be the query sequence with `source_database` set to `"query"` [aligned\_pqt\.py L78-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L78-L80)

### File Naming

 Files are named based on the SHA256 hash of the uppercased query sequence\. The `expected_basename` function handles this mapping [aligned\_pqt\.py L57-L60](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L57-L60)

 Sources: [aligned\_pqt\.py L37-L60](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L37-L60) [data\_source\.py L1-L50](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/data_source.py#L1-L50)

## MSA Generation via ColabFold

 Chai\-1 supports automated MSA generation using the ColabFold server API\. This is primarily handled by `generate_colabfold_msas` [colabfold\.py L296-L302](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L296-L302)

### Implementation Details

 1. **API Interaction**: The function `_run_mmseqs2` \(adapted from ColabFold\) handles submission to `api.colabfold.com` [colabfold\.py L35-L46](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L35-L46) It supports both standard MSA tickets and pairing tickets for multimers [colabfold\.py L48-L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L48-L49)
2. **Tokenization**: Aligned sequences are tokenized into numeric arrays using Numba\-accelerated functions in `a3m.py` [a3m\.py L97-L113](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L97-L113) This process also generates a `deletion_matrix` representing lowercase letters in a3m format [a3m\.py L81-L91](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L81-L91)
3. **Pairing Logic**: For multimers, sequences are paired based on taxid/species information extracted from headers [colabfold\.py L343-L350](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L343-L350)

  Sources: [colabfold\.py L35-L152](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L35-L152) [a3m\.py L57-L113](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L57-L113) [colabfold\.py L296-L458](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L296-L458)

## MSA Integration and Context

 Once loaded, MSAs are encapsulated in the `MSAContext` class, which manages tensors for the model trunk [msa\_context\.py L25-L33](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/msa_context.py#L25-L33)

### MSAContext Structure

| Attribute | Shape | Type | Description |
| --- | --- | --- | --- |
| tokens | \(depth, tokens\) | uint8 | Integer\-encoded amino acids/nucleotides\. |
| pairing\_key\_hash | \(depth, tokens\) | int32 | Hashed pairing keys for cross\-chain alignment\. |
| deletion\_matrix | \(depth, tokens\) | uint8 | Count of deleted residues at each position\. |
| sequence\_source | \(depth, tokens\) | uint8 | Encoded source database ID\. |

### Loading Pipeline

 The `get_msa_contexts` function orchestrates the transition from disk to model\-ready tensors [load\.py L29-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/load.py#L29-L32)

 1. **Discovery**: Locates `.aligned.pqt` files in the provided `msa_directory` using `expected_basename` [load\.py L49-L50](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/load.py#L49-L50)
2. **Parsing**: Uses `parse_aligned_pqt_to_msa_context` to validate schema and apply database quotas [aligned\_pqt\.py L63-L70](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L63-L70)
3. **Pairing & Merging**: - `pair_and_merge_msas` aligns sequences across different chains if they share a pairing key hash [preprocess\.py L86-L89](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L86-L89) - `merge_main_msas_by_chain` concatenates MSAs along the token dimension for multimer featurization [preprocess\.py L23-L24](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L23-L24)
4. **Deduplication**: `drop_duplicates` removes identical rows within the context [preprocess\.py L36-L37](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L36-L37)

  Sources: [load\.py L29-L82](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/load.py#L29-L82) [msa\_context\.py L25-L56](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/msa_context.py#L25-L56) [preprocess\.py L23-L133](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L23-L133)

## Technical Implementation Details

### Tokenization Mapping

 The mapping from a3m characters to model tokens is handled by `_get_tokenization_mapping` [a3m\.py L29-L30](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L29-L30) It handles:

 - **Uppercase**: Mapped to standard residue types [a3m\.py L41-L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L41-L48)
- **Lowercase**: Mapped as insertions \(`MAPPED_TOKEN_INSERTION`\) and counted in the deletion matrix [a3m\.py L50-L51](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L50-L51)
- **Gaps \(`-`\)**: Assigned a specific gap token [a3m\.py L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L52-L52)

### Quotas and Priorities

 During loading, Chai\-1 applies per\-database quotas \(e\.g\., limiting hits from `mgnify` or `uniref90`\) and sorts sequences by priority [aligned\_pqt\.py L83-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L83-L109)

### Pairing Strategy

 Pairing is based on `pairing_key_hash`\. The `prepair_ukey` function generates unique keys `(pair_key, rank)` to handle multiple hits from the same species, ensuring stable pairing across chains [preprocess\.py L49-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L49-L52)

 Sources: [a3m\.py L29-L56](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/a3m.py#L29-L56) [aligned\_pqt\.py L83-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/msas/aligned_pqt.py#L83-L109) [preprocess\.py L49-L83](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/preprocess.py#L49-L83)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments](https://deepwiki.com/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments) on DeepWiki*