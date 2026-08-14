# Template Search

> **Relevant source files**
> * [src/alphafold3/data/parsers.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/parsers.py)
> * [src/alphafold3/data/templates.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py)
> * [src/alphafold3/data/tools/hmmalign.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py)
> * [src/alphafold3/data/tools/hmmbuild.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py)
> * [src/alphafold3/data/tools/hmmsearch.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py)
> * [src/alphafold3/model/confidences.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/model/confidences.py)

Template search is a critical component of the AlphaFold 3 data pipeline that identifies structural templates with similar sequences to the query polymer. These templates provide valuable structural information that guides the model's predictions, particularly for regions with conserved structural features. This document details the implementation and usage of the template search system in AlphaFold 3.

## Overview

The template search process uses `hmmsearch` to identify similar sequences from a structure database, retrieves their 3D coordinates, and transforms them into template features used by the model during inference. Templates significantly improve prediction accuracy by providing real structural examples for the model to learn from.

```

```

Sources: [src/alphafold3/data/templates.py L11-L35](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L11-L35)

 [src/alphafold3/data/tools/hmmsearch.py L11-L20](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L11-L20)

## Key Classes and Components

The template search functionality is implemented primarily through the `Templates` container and the `Hmmsearch` wrapper.

```

```

Sources: [src/alphafold3/data/templates.py L157-L191](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L157-L191)

 [src/alphafold3/data/templates.py L396-L428](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L396-L428)

 [src/alphafold3/data/tools/hmmsearch.py L22-L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L40)

 [src/alphafold3/data/tools/hmmbuild.py L22-L51](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L22-L51)

 [src/alphafold3/data/tools/hmmalign.py L28-L44](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L28-L44)

### Templates Class

The `Templates` class serves as the primary container for template hits matched to a query sequence. It manages collections of hits and provides methods for filtering, featurizing, and accessing template structures.

Key Methods:

* `from_seq_and_a3m`: Creates templates from a query sequence and MSA in A3M format. [src/alphafold3/data/templates.py L430-L484](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L430-L484)
* `from_hmmsearch_a3m`: Creates templates directly from HMMSearch A3M output. [src/alphafold3/data/templates.py L486-L531](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L486-L531)
* `filter`: Filters hits based on configurable criteria such as release date and sequence identity. [src/alphafold3/data/templates.py L599-L642](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L599-L642)
* `featurize`: Transforms the templates into features usable by the model, including atom positions and masks. [src/alphafold3/data/templates.py L665-L718](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L665-L718)

### Hit Class

The `Hit` class represents a single template match with detailed alignment information:

| Field | Description |
| --- | --- |
| `pdb_id` | The PDB ID of the matching structure. |
| `auth_chain_id` | The author chain ID within the structure. |
| `hmmsearch_sequence` | Template sequence with alignment information (gaps/insertions). |
| `structure_sequence` | Template sequence from the 3D structure. |
| `query_sequence` | The original query sequence. |
| `start_index/end_index` | Indices mapping to the full PDB sequence. |
| `release_date` | Structure release date. |
| `chain_poly_type` | Polymer type (protein, RNA, DNA). |

Sources: [src/alphafold3/data/templates.py L157-L191](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L157-L191)

## Template Search Workflow

The template search workflow involves several steps from initial query to final template features:

1. **Profile Generation**: `Hmmsearch` uses `Hmmbuild` to convert the input MSA (A3M/Stockholm) into a profile HMM. [src/alphafold3/data/tools/hmmsearch.py L134-L149](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L134-L149)
2. **HMMSearch Execution**: The profile HMM is searched against a database (e.g., PDB70) using the `hmmsearch` binary. [src/alphafold3/data/tools/hmmsearch.py L97-L132](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L97-L132)
3. **Hit Parsing**: `Templates.from_hmmsearch_a3m` parses the output to extract template hits and their alignments using regex on hit descriptions. [src/alphafold3/data/templates.py L137-L140](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L137-L140)  [src/alphafold3/data/templates.py L486-L531](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L486-L531)
4. **Filtering**: Removes templates based on release dates and sequence identity to prevent leakage. [src/alphafold3/data/templates.py L599-L642](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L599-L642)
5. **Featurization**: Transforms templates into `TemplateFeatures` mapping. [src/alphafold3/data/templates.py L50-L52](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L50-L52)  [src/alphafold3/data/templates.py L665-L718](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L665-L718)

```

```

Sources: [src/alphafold3/data/templates.py L430-L484](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L430-L484)

 [src/alphafold3/data/tools/hmmsearch.py L97-L132](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L97-L132)

 [src/alphafold3/data/tools/hmmbuild.py L70-L87](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L70-L87)

## Template Filtering

Template filtering is a crucial step to ensure that only high-quality, relevant templates are used. The filtering criteria include:

| Filter | Purpose | Implementation |
| --- | --- | --- |
| `release_date_cutoff` | Prevents future knowledge leakage during training or evaluation. | `Hit.release_date <= cutoff`. [src/alphafold3/data/templates.py L609](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L609-L609) |
| `max_subsequence_ratio` | Prevents leakage from exact matches or near-exact matches. | Checks identity ratio. [src/alphafold3/data/templates.py L614](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L614-L614) |
| `min_hit_length` | Ensures meaningful template information is present. | Excludes short hits. [src/alphafold3/data/templates.py L618](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L618-L618) |
| `deduplicate_sequences` | Increases diversity of templates by removing redundancy. | Removes identical template sequences. [src/alphafold3/data/templates.py L633](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L633-L633) |
| `max_hits` | Controls computational cost and model input size. | Limits the total number of templates. [src/alphafold3/data/templates.py L642](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L642-L642) |

Sources: [src/alphafold3/data/templates.py L599-L642](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L599-L642)

## Template Featurization

Template features are extracted and transformed into the format required by the model. The process uses `_POLYMER_FEATURES` and `_LIGAND_FEATURES` definitions to shape the output.

### Polymer Features

The `_POLYMER_FEATURES` mapping defines the expected features for protein, RNA, and DNA:

* `template_aatype`: Residue types encoded as integers using `_encode_restype`. [src/alphafold3/data/templates.py L37](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L37-L37)  [src/alphafold3/data/templates.py L89-L134](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L89-L134)
* `template_all_atom_masks`: Binary masks for valid atom positions (e.g., ATOM37 for protein). [src/alphafold3/data/templates.py L38](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L38-L38)
* `template_all_atom_positions`: 3D coordinates of atoms. [src/alphafold3/data/templates.py L39](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L39-L39)
* `template_domain_names`: PDB/Chain identifiers. [src/alphafold3/data/templates.py L40](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L40-L40)
* `template_release_date`: ISO formatted release dates. [src/alphafold3/data/templates.py L41](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L41-L41)
* `template_sequence`: The raw sequence string. [src/alphafold3/data/templates.py L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L42-L42)

### Ligand Features

If enabled, `_LIGAND_FEATURES` are extracted from the structure:

* `ligand_features`: Mapping containing ligand atom positions and names, typically retrieved from `StructureStore`. [src/alphafold3/data/templates.py L45-L47](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L45-L47)  [src/alphafold3/data/templates.py L820-L884](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L820-L884)

Sources: [src/alphafold3/data/templates.py L36-L47](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L36-L47)

 [src/alphafold3/data/templates.py L820-L884](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/templates.py#L820-L884)

## External Tool Integration

The template search relies on the HMMER suite. Wrappers are provided for:

* **Hmmsearch**: Searches a profile against a sequence database. Supports various E-value and pre-filter settings. [src/alphafold3/data/tools/hmmsearch.py L22-L87](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L22-L87)
* **Hmmbuild**: Constructs HMM profiles from MSAs (A3M or Stockholm). [src/alphafold3/data/tools/hmmbuild.py L22-L87](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L22-L87)
* **Hmmalign**: Aligns sequences to a profile and returns results in A3M (A2M) format. [src/alphafold3/data/tools/hmmalign.py L28-L111](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L28-L111)

These tools use `subprocess_utils` to execute binaries and manage temporary files. [src/alphafold3/data/tools/hmmsearch.py L119-L125](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L119-L125)

 [src/alphafold3/data/tools/hmmbuild.py L135-L141](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L135-L141)

Sources: [src/alphafold3/data/tools/hmmsearch.py L11-L150](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmsearch.py#L11-L150)

 [src/alphafold3/data/tools/hmmbuild.py L11-L147](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmbuild.py#L11-L147)

 [src/alphafold3/data/tools/hmmalign.py L11-L144](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/data/tools/hmmalign.py#L11-L144)