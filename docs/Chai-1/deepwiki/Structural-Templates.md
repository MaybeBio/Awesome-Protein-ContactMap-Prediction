# Structural Templates

> **Relevant source files**
> * [chai_lab/data/dataset/templates/align.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/align.py)
> * [chai_lab/data/dataset/templates/context.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py)
> * [chai_lab/data/dataset/templates/load.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py)
> * [chai_lab/data/io/rcsb.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/io/rcsb.py)
> * [chai_lab/data/parsing/templates/m8.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py)
> * [chai_lab/data/parsing/templates/template_hit.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py)
> * [chai_lab/data/sources/rdkit.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/sources/rdkit.py)
> * [chai_lab/tools/kalign.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/tools/kalign.py)
> * [tests/test_rdkit.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_rdkit.py)

This document describes the template-based structure prediction capabilities within the Chai-1 molecular structure prediction system. Templates are known 3D structures with sequence similarity to the target sequence that can significantly improve prediction accuracy by providing structural information from homologous proteins.

## Template System Overview

Templates in Chai-1 are processed through a pipeline that parses template hits from M8 format files, downloads structures from RCSB, aligns sequences using Kalign, and extracts structural features that inform the diffusion model during structure prediction.

### Template Processing Pipeline

```mermaid
flowchart TD

M8File["M8 Format File<br>Template search results"]
parse_m8_file["parse_m8_file()"]
parse_m8_to_template_hits["parse_m8_to_template_hits()"]
TemplateHit["TemplateHit objects"]
download_cif_file["download_cif_file()"]
gemmi_read["gemmi.read_structure()"]
kalign_query_to_reference["kalign_query_to_reference()"]
get_template_data["get_template_data()"]
LoadedTemplate["LoadedTemplate objects"]
template_restype["template_restype"]
template_pseudo_beta_mask["template_pseudo_beta_mask"]
template_distances["template_pseudo_beta_distances"]
template_unit_vector["template_unit_vector"]
template_backbone_frame_mask["template_backbone_frame_mask"]
TemplateContext["TemplateContext.from_loaded_templates()"]
get_template_context["get_template_context()"]
AllAtomFeatureContext["AllAtomFeatureContext"]

TemplateHit --> download_cif_file
LoadedTemplate --> template_restype
LoadedTemplate --> template_pseudo_beta_mask
LoadedTemplate --> template_distances
LoadedTemplate --> template_unit_vector
LoadedTemplate --> template_backbone_frame_mask
template_restype --> TemplateContext
template_pseudo_beta_mask --> TemplateContext
template_distances --> TemplateContext
template_unit_vector --> TemplateContext
template_backbone_frame_mask --> TemplateContext
TemplateContext --> get_template_context
get_template_context --> AllAtomFeatureContext

subgraph subGraph3 ["Context Assembly"]
    TemplateContext
end

subgraph subGraph2 ["Feature Extraction"]
    template_restype
    template_pseudo_beta_mask
    template_distances
    template_unit_vector
    template_backbone_frame_mask
end

subgraph subGraph1 ["Structure Loading"]
    download_cif_file
    gemmi_read
    kalign_query_to_reference
    get_template_data
    LoadedTemplate
    download_cif_file --> gemmi_read
    gemmi_read --> kalign_query_to_reference
    kalign_query_to_reference --> get_template_data
    get_template_data --> LoadedTemplate
end

subgraph subGraph0 ["Template Discovery"]
    M8File
    parse_m8_file
    parse_m8_to_template_hits
    TemplateHit
    M8File --> parse_m8_file
    parse_m8_file --> parse_m8_to_template_hits
    parse_m8_to_template_hits --> TemplateHit
end
```

Sources: [chai_lab/data/parsing/templates/m8.py L22-L130](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L22-L130)

 [chai_lab/data/dataset/templates/load.py L262-L411](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L262-L411)

 [chai_lab/data/dataset/templates/context.py L334-L420](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L334-L420)

## M8 Format Template Hit Discovery

Template hits are discovered by parsing M8 format files, which contain the results of template search algorithms. The M8 format is a tab-delimited format that includes alignment statistics and coordinates.

### M8 File Parsing

```mermaid
flowchart TD

M8File["M8 Format File<br>Tab-delimited search results"]
parse_m8_file["parse_m8_file()"]
DataFrame["pandas.DataFrame<br>query_id, subject_id, pident,<br>length, mismatch, gapopen,<br>query_start, query_end,<br>subject_start, subject_end,<br>evalue, bitscore, comment"]
iterate_rows["Iterate over rows"]
split_subject["Split subject_id<br>into hit_identifier, hit_chain"]
download_cif_file["download_cif_file()"]
gemmi_chain["gemmi.Chain.get_polymer()"]
kalign_query_to_reference["kalign_query_to_reference()"]
tokenize_sequences["tokenize_sequences_to_arrays()"]
TemplateHit["TemplateHit object"]

DataFrame --> iterate_rows

subgraph subGraph1 ["Template Hit Processing"]
    iterate_rows
    split_subject
    download_cif_file
    gemmi_chain
    kalign_query_to_reference
    tokenize_sequences
    TemplateHit
    iterate_rows --> split_subject
    split_subject --> download_cif_file
    download_cif_file --> gemmi_chain
    gemmi_chain --> kalign_query_to_reference
    kalign_query_to_reference --> tokenize_sequences
    tokenize_sequences --> TemplateHit
end

subgraph subGraph0 ["M8 File Format"]
    M8File
    parse_m8_file
    DataFrame
    M8File --> parse_m8_file
    parse_m8_file --> DataFrame
end
```

The `parse_m8_to_template_hits` function processes each hit by:

1. Extracting PDB ID and chain ID from the `subject_id` field [chai_lab/data/parsing/templates/m8.py L79-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L79-L80)
2. Downloading the CIF file using `download_cif_file` [chai_lab/data/parsing/templates/m8.py L82-L87](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L82-L87)
3. Parsing the structure with `gemmi.read_structure` [chai_lab/data/parsing/templates/m8.py L88](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L88-L88)
4. Aligning the hit sequence to the query using `kalign_query_to_reference` [chai_lab/data/parsing/templates/m8.py L97](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L97-L97)
5. Tokenizing the aligned sequence for model input using `tokenize_sequences_to_arrays` [chai_lab/data/parsing/templates/m8.py L101](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L101-L101)

Sources: [chai_lab/data/parsing/templates/m8.py L22-L130](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/m8.py#L22-L130)

 [chai_lab/data/io/rcsb.py L9-L18](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/io/rcsb.py#L9-L18)

## Template Hit Representation

The `TemplateHit` class represents a match between a query sequence and a known structure discovered from M8 search results. Each hit contains alignment information and indices that map between query and template positions.

```mermaid
classDiagram
    note "Created by parse_m8_to_template_hits()from M8 format search results"
    class TemplateHit {
        +query_pdb_id: str
        +query_sequence: str
        +index: int
        +pdb_id: str
        +chain_id: str
        +hit_start: int
        +hit_end: int
        +hit_tokens: Int32Tensor
        +deletion_matrix: UInt8Tensor
        +query_seq_realigned: str
        +cif_path: Path
        +hit_sequence() : str
        +indices_query() : Int32Tensor
        +indices_hit_within_subregion() : Int32Tensor
        +indices_hit() : Int32Tensor
        +hit_valid_mask() : BoolTensor
        +query_start_end() : tuple
    }
```

### Key Properties and Methods

| Property | Type | Description |
| --- | --- | --- |
| `hit_sequence` | `str` | Amino acid sequence of the template hit [chai_lab/data/parsing/templates/template_hit.py L71-L74](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L71-L74) |
| `indices_query` | `Int32[Tensor, "m"]` | Query residue indices corresponding to hit [chai_lab/data/parsing/templates/template_hit.py L77-L83](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L77-L83) |
| `indices_hit` | `Int32[Tensor, "n"]` | Hit indices within full hit sequence [chai_lab/data/parsing/templates/template_hit.py L109-L111](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L109-L111) |
| `hit_valid_mask` | `Bool[Tensor, "n"]` | Mask indicating valid positions (not gaps) [chai_lab/data/parsing/templates/template_hit.py L114-L129](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L114-L129) |
| `query_start_end` | `tuple[int, int]` | Start and end positions in query sequence [chai_lab/data/parsing/templates/template_hit.py L132-L135](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L132-L135) |

The `indices_hit` property accounts for insertions and deletions in the alignment by using the `deletion_matrix` and `hit_tokens` [chai_lab/data/parsing/templates/template_hit.py L92-L106](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L92-L106)

Sources: [chai_lab/data/parsing/templates/template_hit.py L14-L136](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/templates/template_hit.py#L14-L136)

## Template Loading and Processing

Templates are loaded using the `get_template_data` function, which converts `TemplateHit` objects into `LoadedTemplate` objects containing the actual structural data.

### Template Loading Pipeline

```mermaid
flowchart TD

TemplateHitIterator["Iterator[TemplateHit]"]
check_overlap["Check overlap with<br>query_crop_indices"]
_get_entity_data["_get_entity_data()"]
structure_to_entities_data["structure_to_entities_data()"]
AllAtomEntityData["AllAtomEntityData"]
tokenize_entity["tokenizer.tokenize_entity()"]
AllAtomStructureContext["AllAtomStructureContext"]
drop_unresolved["Drop unresolved residues<br>get_centre_positions_and_mask()"]
strict_subsequence_check["Strict subsequence check<br>Compare sequences"]
LoadedTemplate["LoadedTemplate"]
template_restype["template_restype"]
template_pseudo_beta_mask["template_pseudo_beta_mask"]
template_pseudo_beta_distances["template_pseudo_beta_distances"]
template_backbone_frame_mask["template_backbone_frame_mask"]
template_unit_vector["template_unit_vector"]

_get_entity_data --> structure_to_entities_data
AllAtomStructureContext --> drop_unresolved
LoadedTemplate --> template_restype
LoadedTemplate --> template_pseudo_beta_mask
LoadedTemplate --> template_pseudo_beta_distances
LoadedTemplate --> template_backbone_frame_mask
LoadedTemplate --> template_unit_vector

subgraph subGraph3 ["Feature Extraction"]
    template_restype
    template_pseudo_beta_mask
    template_pseudo_beta_distances
    template_backbone_frame_mask
    template_unit_vector
end

subgraph subGraph2 ["Template Validation"]
    drop_unresolved
    strict_subsequence_check
    LoadedTemplate
    drop_unresolved --> strict_subsequence_check
    strict_subsequence_check --> LoadedTemplate
end

subgraph subGraph1 ["Structure Processing"]
    structure_to_entities_data
    AllAtomEntityData
    tokenize_entity
    AllAtomStructureContext
    structure_to_entities_data --> AllAtomEntityData
    AllAtomEntityData --> tokenize_entity
    tokenize_entity --> AllAtomStructureContext
end

subgraph subGraph0 ["Input Processing"]
    TemplateHitIterator
    check_overlap
    _get_entity_data
    TemplateHitIterator --> check_overlap
    check_overlap --> _get_entity_data
end
```

### LoadedTemplate Class Structure

```mermaid
classDiagram
    class LoadedTemplate {
        +query_crop_indices: Int32Tensor
        +template_hit: TemplateHit
        +template_hit_structure_context: AllAtomStructureContext
        +query_identifier() : str
        +hit_identifier() : str
        +template_hit_indices() : Int32Tensor
        +template_query_match_indices() : Int32Tensor
        +cropped_template_query_match_indices() : Int32Tensor
        +template_restype() : Int32Tensor
        +template_pseudo_beta_mask() : BoolTensor
        +template_pseudo_beta_distances() : FloatTensor
        +template_backbone_frame_mask() : BoolTensor
        +template_unit_vector() : FloatTensor
    }
    class TemplateHit {
    }
    class AllAtomStructureContext {
    }
    LoadedTemplate *-- TemplateHit
    LoadedTemplate *-- AllAtomStructureContext
```

### Key Features and Properties

The `LoadedTemplate` class provides properties that extract features directly for the model:

* `template_restype`: Extracts residue types and handles gaps using `rc.residue_types_with_nucleotides_order["-"]` [chai_lab/data/dataset/templates/load.py L121-L128](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L121-L128)
* `template_pseudo_beta_mask`: Computes the existence mask for reference atoms (e.g., Cβ) [chai_lab/data/dataset/templates/load.py L131-L144](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L131-L144)
* `template_pseudo_beta_distances`: Calculates pairwise distances between reference atoms, filling masked regions with a large value (100.0) [chai_lab/data/dataset/templates/load.py L147-L171](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L147-L171)

Sources: [chai_lab/data/dataset/templates/load.py L58-L232](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L58-L232)

 [chai_lab/data/dataset/templates/load.py L262-L411](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L262-L411)

## Template Alignment with Kalign

Sequence alignment is a critical step in template processing. Chai-1 uses the Kalign tool to align query sequences to template sequences through the `kalign_query_to_reference` function.

### Kalign Alignment Process

```mermaid
flowchart TD

QuerySeq["Query Sequence"]
clean_query["query.upper().replace('-', '')"]
RefSeq["Reference Sequence"]
write_fastas["write_fastas()"]
temp_fasta["Temporary FASTA file"]
kalign_subprocess["subprocess.run(['kalign', '-i', input, '-o', output])"]
read_fasta["read_fasta()"]
KalignAlignment["KalignAlignment"]
query_a3m_line["query_a3m_line property"]
reference_span["reference_span property"]
tokenize_sequences["tokenize_sequences_to_arrays()"]
TemplateHit["TemplateHit creation"]

write_fastas --> temp_fasta
KalignAlignment --> query_a3m_line
KalignAlignment --> reference_span

subgraph subGraph2 ["Alignment Processing"]
    query_a3m_line
    reference_span
    tokenize_sequences
    TemplateHit
    query_a3m_line --> tokenize_sequences
    tokenize_sequences --> TemplateHit
end

subgraph subGraph1 ["Kalign Execution"]
    temp_fasta
    kalign_subprocess
    read_fasta
    KalignAlignment
    temp_fasta --> kalign_subprocess
    kalign_subprocess --> read_fasta
    read_fasta --> KalignAlignment
end

subgraph subGraph0 ["Input Preprocessing"]
    QuerySeq
    clean_query
    RefSeq
    write_fastas
    QuerySeq --> clean_query
    clean_query --> write_fastas
    RefSeq --> write_fastas
end
```

### KalignAlignment Class

```mermaid
classDiagram
    note "Validates alignment:- Same length sequences- No double gaps- Proper A3M formatting"
    class KalignAlignment {
        +reference_aligned: str
        +query_aligned: str
        +query_a3m_line() : str
        +reference_span() : tuple[int, int]
        +post_init()
    }
```

The `query_a3m_line` property converts the alignment to A3M format by marking query insertions as lowercase [chai_lab/tools/kalign.py L35-L42](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/tools/kalign.py#L35-L42)

 The `reference_span` property identifies the coverage range on the reference sequence [chai_lab/tools/kalign.py L44-L57](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/tools/kalign.py#L44-L57)

Sources: [chai_lab/tools/kalign.py L19-L111](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/tools/kalign.py#L19-L111)

## Template Context Assembly

Individual `LoadedTemplate` objects are assembled into a unified `TemplateContext` that can be processed by the model. The `TemplateContext` class handles alignment, padding, and merging of multiple templates.

### TemplateContext Structure

```mermaid
classDiagram
    note "Tensor dimensions:[n_templates, n_tokens][n_templates, n_tokens, n_tokens][n_templates, n_tokens, n_tokens, 3]"
    class TemplateContext {
        +template_restype: Int32Tensor
        +template_pseudo_beta_mask: BoolTensor
        +template_backbone_frame_mask: BoolTensor
        +template_distances: FloatTensor
        +template_unit_vector: FloatTensor
        +num_tokens() : int
        +num_templates() : int
        +num_nonnull_templates() : int
        +template_mask() : BoolTensor
        +to_dict() : dict
        +empty() : TemplateContext
        +index_select() : TemplateContext
        +merge() : TemplateContext
        +pad() : TemplateContext
        +from_loaded_templates() : TemplateContext
    }
```

### Template Feature Extraction

Templates provide several key features that inform the structure prediction:

| Feature | Tensor Shape | Description |
| --- | --- | --- |
| `template_restype` | `[n_templates, n_tokens]` | Amino acid residue types [chai_lab/data/dataset/templates/context.py L46](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L46-L46) |
| `template_pseudo_beta_mask` | `[n_templates, n_tokens]` | Mask for valid Cβ positions [chai_lab/data/dataset/templates/context.py L47](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L47-L47) |
| `template_distances` | `[n_templates, n_tokens, n_tokens]` | Pairwise Cβ distances [chai_lab/data/dataset/templates/context.py L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L49-L49) |
| `template_backbone_frame_mask` | `[n_templates, n_tokens]` | Mask for complete N-Cα-C frames [chai_lab/data/dataset/templates/context.py L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L48-L48) |
| `template_unit_vector` | `[n_templates, n_tokens, n_tokens, 3]` | Unit vectors in backbone frame [chai_lab/data/dataset/templates/context.py L50](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L50-L50) |

### Merging and Alignment

The `merge` method concatenates multiple `TemplateContext` objects along the sequence dimension while properly tiling the 2D distance and unit vector matrices [chai_lab/data/dataset/templates/context.py L119-L180](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L119-L180)

 1D and 2D features are aligned to the global token indices using `align_1d` and `align_2d` [chai_lab/data/dataset/templates/align.py L18-L67](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/align.py#L18-L67)

Sources: [chai_lab/data/dataset/templates/context.py L42-L332](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L42-L332)

 [chai_lab/data/dataset/templates/align.py L1-L68](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/align.py#L1-L68)

## Integration with the Prediction Pipeline

Templates are integrated into the `AllAtomFeatureContext`, which serves as the central data structure for structure prediction in Chai-1.

```mermaid
flowchart TD

MSAContext["MSA Context"]
AllAtomFeatureContext["AllAtomFeatureContext"]
TemplateContext["Template Context"]
ESMContext["ESM Embedding Context"]
StructureContext["Structure Context"]
RestraintContext["Restraint Context"]
ModelComponents["Model Components<br>(Feature Embedder, Token Embedder,<br>Trunk, Diffusion Module)"]
DiffusionResults["Diffusion Results"]

MSAContext --> AllAtomFeatureContext
TemplateContext --> AllAtomFeatureContext
ESMContext --> AllAtomFeatureContext
StructureContext --> AllAtomFeatureContext
RestraintContext --> AllAtomFeatureContext
AllAtomFeatureContext --> ModelComponents
ModelComponents --> DiffusionResults
```

Templates are discovered via M8 search results, downloaded from RCSB [chai_lab/data/io/rcsb.py L9-L18](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/io/rcsb.py#L9-L18)

 and processed into tensors. The `CHAI_TEMPLATE_CIF_FOLDER` environment variable can be used to specify the storage location for downloaded CIF files [chai_lab/data/dataset/templates/context.py L36-L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L36-L38)

Sources: [chai_lab/data/dataset/templates/load.py L262-L411](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/load.py#L262-L411)

 [chai_lab/data/dataset/templates/context.py L36-L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/templates/context.py#L36-L38)