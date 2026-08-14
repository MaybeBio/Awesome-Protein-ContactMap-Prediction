# Template Processing

> **Relevant source files**
> * [alphafold/common/protein.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/protein.py)
> * [alphafold/common/residue_constants.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py)
> * [alphafold/data/mmcif_parsing.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py)
> * [alphafold/data/pipeline.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline.py)
> * [alphafold/data/pipeline_multimer.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline_multimer.py)
> * [alphafold/data/templates.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py)
> * [alphafold/data/tools/hhsearch.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hhsearch.py)
> * [alphafold/data/tools/hmmbuild.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmbuild.py)
> * [alphafold/data/tools/hmmsearch.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmsearch.py)
> * [alphafold/model/tf/protein_features.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/tf/protein_features.py)
> * [alphafold/relax/utils.py](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/relax/utils.py)

## Purpose and Scope

This document describes the template processing subsystem within AlphaFold's data pipeline. Template processing identifies structural templates from the Protein Data Bank (PDB) that are homologous to the query sequence and extracts their structural information for use by the neural network.

The pipeline performs automated structural template search via HHsearch or HMMsearch and handles template feature extraction from mmCIF files, including coordinate mapping and residue-level metadata extraction.

**Sources:** [alphafold/data/templates.py L15](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L15-L15)

 [alphafold/data/mmcif_parsing.py L15](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L15-L15)

## Overview

Template processing consists of two main phases:

1. **Template Search**: Using sequence-structure alignment tools (HHsearch or HMMsearch) to identify protein structures similar to the query sequence.
2. **Structure Extraction**: Parsing mmCIF files to extract atomic coordinates, residue mappings, and metadata for identified templates.

The process is managed by the `DataPipeline` [alphafold/data/pipeline.py L123-L143](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline.py#L123-L143)

 which coordinates the `TemplateSearcher` [alphafold/data/pipeline.py L34](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline.py#L34-L34)

 and the `TemplateHitFeaturizer` [alphafold/data/templates.py L441](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L441-L441)

**Sources:** [alphafold/data/pipeline.py L168-L169](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline.py#L168-L169)

 [alphafold/data/templates.py L88-L95](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L88-L95)

## Template Search Workflow

```

```

**Template Search Workflow**: The query sequence is compared against template databases using either HHsearch (for HMM-HMM comparison) or HMMsearch (for HMM-sequence comparison). The tools produce different output formats, which are parsed into a unified `TemplateHit` representation.

**Sources:** [alphafold/data/tools/hhsearch.py L72-L117](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hhsearch.py#L72-L117)

 [alphafold/data/tools/hmmsearch.py L83-L145](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmsearch.py#L83-L145)

 [alphafold/data/parsers.py L518-L642](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/parsers.py#L518-L642)

### HHsearch and HMMsearch Execution

The `HHSearch` class [alphafold/data/tools/hhsearch.py L29-L31](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hhsearch.py#L29-L31)

 wraps the binary and queries databases using A3M input [alphafold/data/tools/hhsearch.py L72-L109](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hhsearch.py#L72-L109)

The `Hmmsearch` class [alphafold/data/tools/hmmsearch.py L29-L30](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmsearch.py#L29-L30)

 uses `Hmmbuild` [alphafold/data/tools/hmmbuild.py L27-L28](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmbuild.py#L27-L28)

 to construct a profile from a Stockholm MSA before searching the sequence database [alphafold/data/tools/hmmsearch.py L83-L88](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmsearch.py#L83-L88)

**Sources:** [alphafold/data/tools/hhsearch.py L84-L90](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hhsearch.py#L84-L90)

 [alphafold/data/tools/hmmsearch.py L98-L112](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/tools/hmmsearch.py#L98-L112)

## mmCIF File Parsing

After template search identifies candidate structures, their atomic coordinates and metadata are extracted from mmCIF format files.

```

```

**mmCIF Parsing Workflow**: The parsing process uses Biopython to extract the 3D structure and raw mmCIF dictionary, then processes this data to create SEQRES-to-structure mappings.

**Sources:** [alphafold/data/mmcif_parsing.py L170-L214](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L170-L214)

 [alphafold/data/mmcif_parsing.py L254-L290](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L254-L290)

### Core mmCIF Data Structures

| Data Structure | Purpose | Key Fields |
| --- | --- | --- |
| `AtomSite` | Individual atom information | `residue_name`, `author_chain_id`, `mmcif_seq_num` [alphafold/data/mmcif_parsing.py L43-L51](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L43-L51) |
| `ResidueAtPosition` | Maps SEQRES to structure | `position`, `name`, `is_missing` [alphafold/data/mmcif_parsing.py L63-L68](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L63-L68) |
| `MmcifObject` | Complete parsed file | `header`, `structure`, `chain_to_seqres` [alphafold/data/mmcif_parsing.py L71-L94](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L71-L94) |
| `ParsingResult` | Parse operation result | `mmcif_object`, `errors` [alphafold/data/mmcif_parsing.py L97-L108](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L97-L108) |

**Sources:** [alphafold/data/mmcif_parsing.py L34-L108](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L34-L108)

## Residue Constants and Chemical Properties

Template processing relies on `residue_constants.py` for atom definitions and rigid group coordinates.

```

```

**Residue Constants Structure**: The module provides mappings used throughout template processing and coordinate transformations.

**Sources:** [alphafold/common/residue_constants.py L34-L78](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py#L34-L78)

 [alphafold/common/residue_constants.py L143-L163](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py#L143-L163)

 [alphafold/common/residue_constants.py L526-L533](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py#L526-L533)

### Rigid Group Atom Positions

AlphaFold represents atoms relative to 8 rigid groups (0: backbone, 3: psi-group, 4-7: chi1-4 groups). The coordinates in `rigid_group_atom_positions` [alphafold/common/residue_constants.py L143-L200](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py#L143-L200)

 define the geometry used to reconstruct the full protein structure from frame rotations.

**Sources:** [alphafold/common/residue_constants.py L131-L142](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/common/residue_constants.py#L131-L142)

## Template Feature Integration

The `TemplateHitFeaturizer` processes `TemplateHit` objects to generate numerical features.

### Feature Definitions

The final template features used by the model include:

* `template_aatype`: One-hot encoded amino acid types [alphafold/data/templates.py L89](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L89-L89)
* `template_all_atom_masks`: Binary mask for atom presence [alphafold/data/templates.py L90](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L90-L90)
* `template_all_atom_positions`: XYZ coordinates of all 37 atom types [alphafold/data/templates.py L91](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L91-L91)
* `template_sum_probs`: HHsearch sum of probabilities [alphafold/data/templates.py L94](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L94-L94)

**Sources:** [alphafold/data/templates.py L88-L95](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L88-L95)

 [alphafold/model/tf/protein_features.py L58-L68](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/model/tf/protein_features.py#L58-L68)

### Prefiltering Logic

Before parsing mmCIF files, AlphaFold applies prefilters to template hits:

1. **Date Cutoff**: Excludes templates released after a specified date [alphafold/data/templates.py L108-L126](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L108-L126)
2. **Alignment Ratio**: Excludes hits with low overlap to the query [alphafold/data/templates.py L76-L77](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L76-L77)
3. **Redundancy**: Excludes hits that are exact subsequences of the query [alphafold/data/templates.py L80-L81](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L80-L81)

**Sources:** [alphafold/data/templates.py L175-L208](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L175-L208)

## Multimer Template Processing

For multimer models, monomer template features are reshaped and converted:

* `template_aatype` is remapped from HHsearch order to AlphaFold's internal order [alphafold/data/pipeline_multimer.py L100-L101](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline_multimer.py#L100-L101)
* `template_all_atom_masks` is renamed to `template_all_atom_mask` [alphafold/data/pipeline_multimer.py L102-L103](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline_multimer.py#L102-L103)
* Entity and symmetry IDs are added to distinguish between chains in homomers and heteromers [alphafold/data/pipeline_multimer.py L130-L167](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline_multimer.py#L130-L167)

**Sources:** [alphafold/data/pipeline_multimer.py L79-L106](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline_multimer.py#L79-L106)

## Summary

Template processing provides structural context to the model by:

1. Identifying homologs via **HHsearch** or **HMMsearch**.
2. Parsing structural data from **mmCIF** files into a standardized `MmcifObject`.
3. Filtering hits based on **release date** and **alignment quality**.
4. Mapping coordinates to the **atom37** representation using **residue constants**.

**Sources:** [alphafold/data/templates.py L15-L24](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/templates.py#L15-L24)

 [alphafold/data/pipeline.py L123-L173](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/pipeline.py#L123-L173)

 [alphafold/data/mmcif_parsing.py L71-L108](https://github.com/google-deepmind/alphafold/blob/c77e5d2a/alphafold/data/mmcif_parsing.py#L71-L108)