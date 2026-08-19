# mmCIF and Template Processing

> **Relevant source files**
> * [scripts/get_assembly_from_mmcif.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py)
> * [scripts/get_chain_mapper_from_mmcif.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_chain_mapper_from_mmcif.py)
> * [scripts/get_pdb_assembly.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_pdb_assembly.py)
> * [unifold/msa/SCOPData.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/SCOPData.py)
> * [unifold/msa/mmcif.py](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py)

This page covers the mmCIF (macromolecular Crystallographic Information File) processing system in Uni-Fold, which handles structural template data from the Protein Data Bank to enhance protein structure predictions. The system parses mmCIF files, extracts assembly information, and prepares structural templates for integration into the prediction pipeline.

For information about MSA generation and homology search, see [Homology Search and MSA Generation](/dptech-corp/Uni-Fold/4.1-homology-search-and-msa-generation). For details about how processed templates are used in the model architecture, see [Template Processing](/dptech-corp/Uni-Fold/5.4-template-processing).

## mmCIF Processing Pipeline

The mmCIF processing system follows a multi-stage pipeline that converts raw structural data into model-ready features:

```mermaid
flowchart TD

A["mmCIF Files<br>(*.cif, *.cif.gz)"]
B["MMCIFParser<br>(BioPython)"]
C["parse()<br>unifold.msa.mmcif"]
D["MmcifObject<br>Data Structure"]
E["Chain Validation<br>_get_protein_chains()"]
F["Sequence Extraction<br>chain_to_seqres"]
G["Structure Mapping<br>seqres_to_structure"]
H["Chain ID Mapping<br>mmcif_to_author_chain_id"]
I["Template Features"]
J["Assembly Processing<br>get_assembly_from_mmcif.py"]
K["Rotation Matrices<br>_pdbx_struct_oper_list"]
L["Translation Vectors<br>_pdbx_struct_oper_list"]
M["Chain Operations<br>Assembly Metadata"]
N["Structural Templates<br>for Model Input"]

A --> B
B --> C
C --> D
D --> E
E --> F
E --> G
E --> H
F --> I
G --> I
H --> I
D --> J
J --> K
J --> L
J --> M
K --> N
L --> N
M --> N
I --> N
```

**mmCIF Processing Pipeline**

Sources: [unifold/msa/mmcif.py L170-L359](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L170-L359)

 [scripts/get_assembly_from_mmcif.py L112-L224](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L112-L224)

## Core Data Structures

The mmCIF processing system uses several key data structures to represent parsed structural information:

```mermaid
classDiagram
    class MmcifObject {
        +file_id: str
        +header: PdbHeader
        +structure: PdbStructure
        +chain_to_seqres: Mapping[ChainId, SeqRes]
        +seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
        +mmcif_to_author_chain_id: Mapping[ChainId, ChainId]
        +valid_chains: Mapping[ChainId, str]
    }
    class AtomSite {
        +residue_name: str
        +author_chain_id: str
        +mmcif_chain_id: str
        +author_seq_num: str
        +mmcif_seq_num: int
        +insertion_code: str
        +hetatm_atom: str
        +model_num: int
    }
    class ResidueAtPosition {
        +position: Optional[ResiduePosition]
        +name: str
        +is_missing: bool
        +hetflag: str
    }
    class ResiduePosition {
        +chain_id: str
        +residue_number: int
        +insertion_code: str
    }
    class ParsingResult {
        +mmcif_object: Optional[MmcifObject]
        +errors: Mapping[Tuple[str, str], Any]
    }
    MmcifObject --> ResidueAtPosition : contains
    ResidueAtPosition --> ResiduePosition : has
    ParsingResult --> MmcifObject : contains
    AtomSite --> MmcifObject : parsed_into
```

**mmCIF Data Structure Relationships**

Sources: [unifold/msa/mmcif.py L35-L109](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L35-L109)

## mmCIF Parsing Functions

The system provides two main parsing entry points with different levels of detail:

| Function | Purpose | Structure Data | Performance |
| --- | --- | --- | --- |
| `parse()` | Full parsing with BioPython structure | Complete | Slower |
| `fast_parse()` | Lightweight parsing without structure | Limited | Faster |

Both functions use the same core parsing logic but differ in their output completeness:

```mermaid
flowchart TD

A["mmcif_string"]
B["MMCIFParser.get_structure()"]
C["_mmcif_dict extraction"]
D["_get_header()"]
E["_get_protein_chains()"]
F["_get_atom_site_list()"]
G["parse()"]
H["fast_parse()"]
I["Full MmcifObject<br>with structure"]
J["Lightweight MmcifObject<br>no structure"]

A --> B
B --> C
C --> D
C --> E
C --> F
D --> G
E --> G
F --> G
D --> H
E --> H
G --> I
H --> J
```

**Parsing Function Workflow**

Sources: [unifold/msa/mmcif.py L170-L234](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L170-L234)

 [unifold/msa/mmcif.py L236-L359](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L236-L359)

## Assembly Information Extraction

The assembly processing system extracts biological assembly information from mmCIF files, including symmetry operations and chain transformations:

```mermaid
flowchart TD

A["mmCIF Assembly Sections"]
B["_pdbx_struct_assembly"]
C["_pdbx_struct_assembly_gen"]
D["_pdbx_struct_oper_list"]
E["oligomeric_count<br>Assembly size"]
F["assembly_id<br>Assembly identifier"]
G["oper_expression<br>Operation indices"]
H["asym_id_list<br>Chain lists"]
I["Rotation Matrix<br>9 elements"]
J["Translation Vector<br>3 elements"]
K["process_block_to_dict()"]
L["get_transform()"]
M["Assembly Metadata"]
N["Chain Operations<br>List[Tuple[chains, transform]]"]

A --> B
A --> C
A --> D
B --> E
C --> F
C --> G
C --> H
D --> I
D --> J
E --> K
F --> K
G --> K
H --> K
I --> L
J --> L
K --> M
L --> M
M --> N
```

**Assembly Processing Workflow**

The system handles various assembly formats including ranges (`"1-4"`) and comma-separated lists (`"1,3,5"`):

Sources: [scripts/get_assembly_from_mmcif.py L80-L97](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L80-L97)

 [scripts/get_assembly_from_mmcif.py L185-L220](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L185-L220)

## Chain ID Mapping System

mmCIF files use different chain ID systems that must be mapped for consistency:

| ID Type | Description | Usage |
| --- | --- | --- |
| `mmcif_chain_id` | Internal mmCIF identifier | Parsing and validation |
| `author_chain_id` | Author-assigned identifier | Final output and PDB compatibility |

```mermaid
flowchart TD

A["mmCIF Chain IDs<br>(A, B, C...)"]
B["mmcif_to_author_chain_id<br>Mapping Dictionary"]
C["Author Chain IDs<br>(A, B, C...)"]
D["_get_atom_site_list()"]
E["_atom_site.label_asym_id"]
F["_atom_site.auth_asym_id"]
G["Template Features<br>for Model Input"]
H["Valid Chains<br>Protein only"]
I["Chain Validation<br>_get_protein_chains()"]

A --> B
B --> C
D --> B
E --> D
F --> D
C --> G
H --> I
I --> B
```

**Chain ID Mapping Process**

Sources: [unifold/msa/mmcif.py L209-L226](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L209-L226)

 [unifold/msa/mmcif.py L286-L293](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L286-L293)

## Protein Chain Validation

The system validates and filters chains to ensure only protein chains are processed:

```mermaid
flowchart TD

A["Entity Poly Sequences<br>_entity_poly_seq"]
B["Chemical Components<br>_chem_comp"]
C["Structure Asymmetry<br>_struct_asym"]
D["Monomer Extraction<br>entity_id -> [Monomer]"]
E["Peptide Validation<br>peptide in _chem_comp.type"]
F["Chain Mapping<br>entity_id -> chain_ids"]
G["polymers<br>defaultdict"]
H["Protein Filter"]
I["valid_chains<br>Dict[ChainId, Sequence[Monomer]]"]
J["Minimum Length Check<br>MIN_LENGTH = 21"]
K["Final Valid Chains"]

A --> B
B --> C
A --> D
B --> E
C --> F
D --> G
E --> H
F --> H
G --> H
H --> I
I --> J
J --> K
```

**Protein Chain Validation Process**

Sources: [unifold/msa/mmcif.py L427-L478](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L427-L478)

 [unifold/msa/mmcif.py L367](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L367-L367)

## Sequence and Structure Mapping

The system creates bidirectional mappings between sequence positions and structural coordinates:

| Mapping Type | Source | Target | Purpose |
| --- | --- | --- | --- |
| `chain_to_seqres` | Chain ID | 1-letter sequence | Sequence alignment |
| `seqres_to_structure` | Sequence index | `ResidueAtPosition` | Structure lookup |

```mermaid
flowchart TD

A["SEQRES Records<br>Sequence data"]
B["3-letter codes<br>monomer.id"]
C["SCOPData.protein_letters_3to1"]
D["1-letter codes<br>amino acid sequence"]
E["ATOM Records<br>Coordinate data"]
F["Residue Positions<br>author_seq_num"]
G["ResidueAtPosition<br>with coordinates"]
H["chain_to_seqres<br>Dict[str, str]"]
I["seqres_to_structure<br>Dict[str, Dict[int, ResidueAtPosition]]"]
J["Template Features"]

A --> B
B --> C
C --> D
E --> F
F --> G
D --> H
G --> I
H --> J
I --> J
```

**Sequence-Structure Mapping**

Sources: [unifold/msa/mmcif.py L333-L341](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L333-L341)

 [unifold/msa/mmcif.py L313-L321](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L313-L321)

 [unifold/msa/SCOPData.py L22-L282](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/SCOPData.py#L22-L282)

## Error Handling and Validation

The mmCIF processing system includes comprehensive error handling for common parsing issues:

| Error Type | Detection | Handling |
| --- | --- | --- |
| No protein chains | Chain validation | Return empty result |
| Parse errors | BioPython exceptions | Log and continue |
| Resolution filtering | Header analysis | Skip low-quality structures |
| Invalid sequences | Frequency analysis | Filter homopolymers |

```mermaid
flowchart TD

A["mmCIF Input"]
B["try/catch Block"]
C["BioPython Parsing"]
D["Parse Success?"]
E["Log Error<br>Return ParsingResult with errors"]
F["Protein Chain Check"]
G["Protein Chains Found?"]
H["Return 'No protein chains'"]
I["Resolution Check"]
J["Resolution < 9Å?"]
K["Return 'resolution' error"]
L["Sequence Validation"]
M["Max AA freq < 80%?"]
N["Filter invalid chains"]
O["Success Processing"]
P["ParsingResult"]

A --> B
B --> C
C --> D
D --> E
D --> F
F --> G
G --> H
G --> I
I --> J
J --> K
J --> L
L --> M
M --> N
M --> O
E --> P
H --> P
K --> P
N --> P
O --> P
```

**Error Handling Workflow**

Sources: [scripts/get_assembly_from_mmcif.py L117-L131](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L117-L131)

 [scripts/get_assembly_from_mmcif.py L144-L148](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L144-L148)

 [unifold/msa/mmcif.py L229-L233](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L229-L233)

## Integration with Uni-Fold Pipeline

The processed mmCIF data integrates into the broader Uni-Fold system through several pathways:

```mermaid
flowchart TD

A["mmCIF Processing"]
B["Template Features"]
C["UnifoldDataset<br>Data Loading"]
D["TemplateEmbedders<br>Model Input"]
E["Assembly Metadata"]
F["UF-Symmetry<br>Symmetric Complexes"]
G["Chain Mappings"]
H["Multimer Pipeline<br>Complex Prediction"]
I["AlphaFold Model<br>Structure Prediction"]

A --> B
B --> C
C --> D
A --> E
E --> F
A --> G
G --> H
D --> I
F --> I
H --> I
```

**mmCIF Integration Points**

Sources: [scripts/get_assembly_from_mmcif.py L246-L260](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/scripts/get_assembly_from_mmcif.py#L246-L260)

 [unifold/msa/mmcif.py L343-L353](https://github.com/dptech-corp/Uni-Fold/blob/7b55789e/unifold/msa/mmcif.py#L343-L353)