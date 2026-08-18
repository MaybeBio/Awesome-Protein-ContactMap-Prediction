---
title: "Data Loading and Parsing"
source: deepwiki.com
owner: baker-laboratory
repo: RoseTTAFold-All-Atom
url: https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.3-data-loading-and-parsing
---
# Data Loading and Parsing

# Data Loading and Parsing

> **Relevant source files**
> - [rf2aa/data/data\_loader\_utils\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py)
> - [rf2aa/data/parsers\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py)
> - [rf2aa/data/protein\.py](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py)

 This page documents the data loading and parsing subsystem of RoseTTAFold All\-Atom \(RFAA\), which is responsible for processing various input files into standardized formats suitable for structure prediction\. The subsystem handles multiple types of biomolecular data including proteins, nucleic acids, and small molecules, converting raw inputs into feature representations required by the neural network models\.

 For information about how inputs are merged after loading, see [Input Merging](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.2-input-merging)\. For details on the inference pipeline that uses these parsed inputs, see [Inference Pipeline](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.4-inference-pipeline)\.

## System Overview

 The data loading and parsing system converts various biological data formats into structured representations that can be used by the RFAA neural network\. This process involves several key stages:

```mermaid
flowchart TD

FASTA["FASTA Files (.fasta)"]
A3M["MSA Files (.a3m)"]
PDB["Structure Files (.pdb)"]
SDF["Small Molecule Files (.sdf/.mol2)"]
PARSE_FASTA["parse_fasta/parse_multichain_fasta"]
PARSE_A3M["parse_a3m"]
PARSE_PDB["parse_pdb"]
PARSE_MOL["parse_mol"]
MSA_FEAT["MSAFeaturize"]
TEMPL_FEAT["TemplFeaturize"]
BOND_FEAT["get_bond_features"]
RAW_DATA["RawInputData"]

FASTA --> PARSE_FASTA
A3M --> PARSE_A3M
PDB --> PARSE_PDB
SDF --> PARSE_MOL
PARSE_FASTA --> MSA_FEAT
PARSE_A3M --> MSA_FEAT
PARSE_PDB --> TEMPL_FEAT
PARSE_MOL --> BOND_FEAT
MSA_FEAT --> RAW_DATA
TEMPL_FEAT --> RAW_DATA
BOND_FEAT --> RAW_DATA

subgraph subGraph3 ["Data Structure"]
    RAW_DATA
end

subgraph Featurization ["Featurization"]
    MSA_FEAT
    TEMPL_FEAT
    BOND_FEAT
end

subgraph subGraph1 ["Parsing Functions"]
    PARSE_FASTA
    PARSE_A3M
    PARSE_PDB
    PARSE_MOL
end

subgraph subGraph0 ["Input Files"]
    FASTA
    A3M
    PDB
    SDF
end
```

 Sources: [parsers\.py L1-L813](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L1-L813) [data\_loader\_utils\.py L1-L910](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L1-L910) [protein\.py L1-L94](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py#L1-L94)

## File Format Parsing

 RFAA supports several standard biological file formats, each parsed by specialized functions to extract relevant information\.

### A3M Files \(Multiple Sequence Alignments\)

 A3M files contain multiple sequence alignments \(MSAs\) which provide evolutionary information crucial for structure prediction\. The `parse_a3m` function handles these files:

```mermaid
flowchart TD

A3M["A3M File (.a3m/.a3m.gz)"]
PARSE["parse_a3m()"]
MSA["MSA Matrix (N_seq × L)"]
INS["Insertion Matrix (N_seq × L)"]
TAXID["Taxonomic IDs (N_seq)"]
AA_MAP["AA → Numerical Index 0-20"]
INS_TRACK["Track Insertions (0 = match, >0 = ins length)"]
TAX_EXTRACT["Extract TaxID from Headers"]

A3M --> PARSE
PARSE --> MSA
PARSE --> INS
PARSE --> TAXID
PARSE --> AA_MAP
PARSE --> INS_TRACK
PARSE --> TAX_EXTRACT

subgraph subGraph1 ["Conversion Details"]
    AA_MAP
    INS_TRACK
    TAX_EXTRACT
end

subgraph subGraph0 ["MSA Contents"]
    MSA
    INS
    TAXID
end
```

 Sources: [parsers\.py L401-L480](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L401-L480)

### PDB Files \(3D Structures\)

 PDB files contain 3D structural coordinates used primarily for template information\. The `parse_pdb` function extracts atom positions and other structural details:

```mermaid
flowchart TD

PDB["PDB File (.pdb)"]
PARSE["parse_pdb()"]
XYZ["Coordinates (L × N_atoms × 3)"]
MASK["Atom Mask (L × N_atoms)"]
IDX["Residue Indices (L)"]
SEQ["Sequence (L)"]
LDDT["pLDDT Mask"]
ATOMS["Extract CA, P, etc."]
COORDS["Store xyz Coordinates"]
MISSING["Flag Missing Atoms"]

PDB --> PARSE
PARSE --> XYZ
PARSE --> MASK
PARSE --> IDX
PARSE --> SEQ
PARSE --> LDDT
PARSE --> ATOMS
PARSE --> COORDS
PARSE --> MISSING

subgraph subGraph1 ["Atom Handling"]
    ATOMS
    COORDS
    MISSING
end

subgraph subGraph0 ["Optional Outputs"]
    SEQ
    LDDT
end
```

 Sources: [parsers\.py L485-L548](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L485-L548)

### FASTA Files \(Sequences\)

 FASTA files provide the primary sequence information\. RFAA includes several parsing functions for different FASTA variants:

 - `parse_fasta`: Basic FASTA parser
- `parse_multichain_fasta`: Handles multiple chains separated by '/'
- `parse_mixed_fasta`: Processes protein/nucleic acid combinations

```mermaid
flowchart TD

FASTA["FASTA File (.fasta)"]
PARSER["Parser Selection"]
SINGLE["parse_fasta()"]
MULTI["parse_multichain_fasta()"]
MIXED["parse_mixed_fasta()"]
MSA_BASIC["MSA Matrix<br>Insertion Matrix"]
MSA_CHAINS["MSA Matrix<br>Insertion Matrix<br>Chain Lengths"]
MSA_MIXED["Protein MSA<br>RNA MSA<br>Insertion Matrix"]
PROT["Protein Alphabet (ARNDCQEGHILKMFPSTWYV)"]
RNA["RNA Alphabet (ACGUN)"]
DNA["DNA Alphabet (ACGTD)"]

FASTA --> PARSER
PARSER --> SINGLE
PARSER --> MULTI
PARSER --> MIXED
SINGLE --> MSA_BASIC
MULTI --> MSA_CHAINS
MIXED --> MSA_MIXED
SINGLE --> PROT
MULTI --> PROT
MULTI --> RNA
MULTI --> DNA
MIXED --> PROT
MIXED --> RNA

subgraph subGraph0 ["Alphabet Selection"]
    PROT
    RNA
    DNA
end
```

 Sources: [parsers\.py L152-L313](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L152-L313)

### Small Molecule Files \(SDF/MOL2\)

 Small molecule structures are parsed from SDF or MOL2 files using the `parse_mol` function, which leverages OpenBabel for chemical structure handling:

```mermaid
flowchart TD

SDF["SDF/MOL2 File"]
PARSE["parse_mol()"]
OBMOL["OpenBabel Molecule"]
MSA["Atom Type Sequence"]
INS["Insertion Features (zeros)"]
COORDS["Atom Coordinates (N_sym × N_atoms × 3)"]
MASK["Atom Masks (N_sym × N_atoms)"]
REMOVE_H["Remove Hydrogens"]
AUTOMORPHS["Find Automorphs (Symmetry)"]
CONFORMER["Generate 3D Conformer"]

SDF --> PARSE
PARSE --> OBMOL
PARSE --> MSA
PARSE --> INS
PARSE --> COORDS
PARSE --> MASK
PARSE --> REMOVE_H
PARSE --> AUTOMORPHS
PARSE --> CONFORMER

subgraph subGraph0 ["Optional Processing"]
    REMOVE_H
    AUTOMORPHS
    CONFORMER
end
```

 Sources: [parsers\.py L744-L812](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L744-L812)

## MSA Processing and Featurization

 Once MSAs are parsed from files, they undergo extensive processing to extract features for the neural network model\.

### MSA Featurization

 The `MSAFeaturize` function transforms raw MSA data into feature representations:

```mermaid
flowchart TD

MSA["MSA Matrix (N_seq × L)"]
FEAT["MSAFeaturize()"]
INS["Insertion Matrix (N_seq × L)"]
OUTPUT["Featurized MSA Outputs"]
CLUSTER["Random Cluster Selection"]
MASK["Random Masking (15%)"]
PROFILE["MSA Profile Calculation"]
INS_FEAT["Insertion Feature Calculation"]
TERM["N/C Terminal Flagging"]
SEQ["Masked Sequences (B × L)"]
MSA_C["Clustered MSA (B × N_clust × L × F_seed)"]
MSA_E["Extra MSA (B × N_extra × L × F_extra)"]
MSA_M["Mask Positions (B × N_clust × L)"]

MSA --> FEAT
INS --> FEAT
FEAT --> OUTPUT
FEAT --> CLUSTER
FEAT --> MASK
FEAT --> PROFILE
FEAT --> INS_FEAT
FEAT --> TERM
OUTPUT --> SEQ
OUTPUT --> MSA_C
OUTPUT --> MSA_E
OUTPUT --> MSA_M

subgraph subGraph1 ["Output Features"]
    SEQ
    MSA_C
    MSA_E
    MSA_M
end

subgraph subGraph0 ["MSA Processing Steps"]
    CLUSTER
    MASK
    PROFILE
    INS_FEAT
    TERM
end
```

 Sources: [data\_loader\_utils\.py L55-L246](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L55-L246)

### MSA Merging

 For multi\-chain proteins, MSAs for individual chains need to be merged\. RFAA provides specialized functions:

 - `merge_a3m_hetero`: Merges MSAs from different proteins \(heteromers\)
- `merge_a3m_homo`: Handles repeated units in homo\-oligomers
- `join_msas_by_taxid`: Pairs sequences in different MSAs by taxonomic ID

```mermaid
flowchart TD

CHAINS["Chain IDs"]
MULTI["load_multi_msa()"]
HASH["MSA Hashes"]
TAXID["TaxIDs"]
MINIMAL["load_minimal_multi_msa()"]
EXPAND["expand_multi_msa()"]
MSA_FINAL["Final Multi-Chain MSA"]
MSA_ORIG["Original MSA"]
HOMO["merge_a3m_homo()"]
NMER["Oligomer Count"]
MODE["Merge Mode<br>(repeat/diag/default)"]
MSA_REPEAT["Repeated MSA"]
MSA_A["MSA Chain A"]
HETERO["merge_a3m_hetero()"]
MSA_B["MSA Chain B"]
L_S["Chain Lengths"]
MSA_AB["Combined MSA"]

subgraph subGraph2 ["Multi-Chain MSA Loading"]
    CHAINS
    MULTI
    HASH
    TAXID
    MINIMAL
    EXPAND
    MSA_FINAL
    CHAINS --> MULTI
    HASH --> MULTI
    TAXID --> MULTI
    MULTI --> MINIMAL
    MULTI --> EXPAND
    MINIMAL --> EXPAND
    EXPAND --> MSA_FINAL
end

subgraph subGraph1 ["Homomeric Merging"]
    MSA_ORIG
    HOMO
    NMER
    MODE
    MSA_REPEAT
    MSA_ORIG --> HOMO
    NMER --> HOMO
    MODE --> HOMO
    HOMO --> MSA_REPEAT
end

subgraph subGraph0 ["Heteromeric Merging"]
    MSA_A
    HETERO
    MSA_B
    L_S
    MSA_AB
    MSA_A --> HETERO
    MSA_B --> HETERO
    L_S --> HETERO
    HETERO --> MSA_AB
end
```

 Sources: [data\_loader\_utils\.py L370-L458](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L370-L458) [data\_loader\_utils\.py L506-L840](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L506-L840)

## Template Processing

 Templates provide structural information that guides the prediction process\. RFAA includes functions for processing template structures:

### Template Featurization

 The `TemplFeaturize` function converts template structures into features for the model:

```mermaid
flowchart TD

TPLT["Template Data"]
FEAT["TemplFeaturize()"]
PARAMS["Parameters"]
XYZ["Coordinates (N_templ × L × N_atoms × 3)"]
T1D["1D Features (N_templ × L × F_1d)"]
MASK["Atom Masks (N_templ × L × N_atoms)"]
IDS["Template IDs"]
SEQ_CUT["Filter by Sequence Identity"]
PICK["Random or Top Selection"]
EXTRACT["Extract Aligned Regions"]
ALIGN["Map to Query Sequence"]
BLANK["blank_template()"]
RANDOM["Random Initialization"]

TPLT --> FEAT
PARAMS --> FEAT
FEAT --> XYZ
FEAT --> T1D
FEAT --> MASK
FEAT --> IDS
FEAT --> SEQ_CUT
FEAT --> PICK
FEAT --> EXTRACT
FEAT --> ALIGN

subgraph subGraph1 ["Alternative: No Templates"]
    XYZ
    BLANK
    RANDOM
    BLANK --> RANDOM
    BLANK --> XYZ
end

subgraph subGraph0 ["Template Selection Steps"]
    SEQ_CUT
    PICK
    EXTRACT
    ALIGN
end
```

 Sources: [data\_loader\_utils\.py L264-L319](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L264-L319) [data\_loader\_utils\.py L248-L261](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L248-L261)

### Template Parsing

 Template information is typically extracted from HHR/ATAB files generated by HHsearch:

```mermaid
flowchart TD

HHR["HHR File"]
PARSE["parse_templates_raw()"]
ATAB["ATAB File"]
FFDB["Template Database"]
XYZ["Template Coordinates"]
MASK["Atom Masks"]
QMAP["Query-Template Mapping"]
F0D["0D Features (probabilities, etc.)"]
F1D["1D Features (per-position scores)"]
SEQ["Template Sequences"]
IDS["Template IDs"]
GET["get_templates()"]
TEMPL_FEAT["TemplFeaturize()"]

HHR --> PARSE
ATAB --> PARSE
FFDB --> PARSE
PARSE --> XYZ
PARSE --> MASK
PARSE --> QMAP
PARSE --> F0D
PARSE --> F1D
PARSE --> SEQ
PARSE --> IDS
GET --> TEMPL_FEAT
PARSE --> GET
```

 Sources: [parsers\.py L624-L726](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L624-L726) [protein\.py L10-L52](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py#L10-L52)

## Data Structure Integration

 The parsed and processed data is integrated into a `RawInputData` structure that is used by the rest of the system:

```mermaid
classDiagram
    class RawInputData {
        msa: torch.Tensor
        ins: torch.Tensor
        bond_features: torch.Tensor
        template_coords: torch.Tensor
        template_mask: torch.Tensor
        template_features: torch.Tensor
        chiral_restraints: torch.Tensor
        atom_frames: torch.Tensor
        taxids: np.array
    }
    class load_protein {
        +msa_file: str
        +hhr_fn: str
        +atab_fn: str
        +model_runner: ModelRunner
        +return: RawInputData
    }
    class generate_msa_and_load_protein {
        +fasta_file: str
        +chain: str
        +model_runner: ModelRunner
        +return: RawInputData
    }
    load_protein --> RawInputData : creates
    generate_msa_and_load_protein --> load_protein : calls
    generate_msa_and_load_protein --> RawInputData : returns
```

 Sources: [protein\.py L55-L94](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py#L55-L94)

## Data Transformation Flow

 The complete data flow from input files to model\-ready features follows this path:

```mermaid
flowchart TD

FASTA["FASTA Files"]
A3M["A3M Files"]
PDB["PDB Files"]
SDF["SDF/MOL2 Files"]
PARSE_FASTA["parse_fasta()"]
PARSE_A3M["parse_a3m()"]
PARSE_PDB["parse_pdb()"]
PARSE_MOL["parse_mol()"]
MSA_FEAT["MSAFeaturize()"]
TEMPL_FEAT["TemplFeaturize()"]
BOND_FEAT["get_bond_features()"]
RAW_DATA["RawInputData"]
RF_INPUT["RFInput"]

FASTA --> PARSE_FASTA
PARSE_FASTA --> MSA_FEAT
A3M --> PARSE_A3M
PARSE_A3M --> MSA_FEAT
PDB --> PARSE_PDB
PARSE_PDB --> TEMPL_FEAT
PARSE_A3M --> BOND_FEAT
SDF --> PARSE_MOL
PARSE_MOL --> BOND_FEAT
MSA_FEAT --> RAW_DATA
TEMPL_FEAT --> RAW_DATA
BOND_FEAT --> RAW_DATA
RAW_DATA --> RF_INPUT

subgraph subGraph4 ["Model Input"]
    RF_INPUT
end

subgraph subGraph3 ["Data Structure"]
    RAW_DATA
end

subgraph subGraph2 ["Processing Layer"]
    MSA_FEAT
    TEMPL_FEAT
    BOND_FEAT
end

subgraph subGraph1 ["Parsing Layer"]
    PARSE_FASTA
    PARSE_A3M
    PARSE_PDB
    PARSE_MOL
end

subgraph subGraph0 ["Input Files"]
    FASTA
    A3M
    PDB
    SDF
end
```

 Sources: [parsers\.py L1-L813](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L1-L813) [data\_loader\_utils\.py L1-L910](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L1-L910) [protein\.py L1-L94](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py#L1-L94)

## Key Implementation Details

### A3M Parsing

 The `parse_a3m` function handles the extraction of MSA information from A3M files:

 1. Reads the file line by line \(supporting gzipped files\)
2. Extracts taxonomic IDs from headers
3. Processes sequence lines and records insertion information
4. Converts amino acid letters to numeric indices \(0\-20\)
5. Returns MSA matrix, insertion matrix, and taxonomic IDs

 Sources: [parsers\.py L401-L480](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L401-L480)

### MSA Featurization

 The `MSAFeaturize` function transforms raw MSA data into features:

 1. Takes MSA and insertion matrices as input
2. Performs block deletion for data augmentation
3. Clusters sequences and applies random masking
4. Calculates profiles and insertion statistics
5. Returns multiple feature representations for the model

 Sources: [data\_loader\_utils\.py L55-L246](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L55-L246)

### Template Processing

 Template processing involves:

 1. Parsing template files \(HHR/ATAB\) to extract alignments and scores
2. Selecting templates based on sequence identity and coverage
3. Extracting coordinates, masks, and features from template structures
4. Mapping template residues to query sequence positions
5. Creating blank templates when no suitable templates are available

 Sources: [parsers\.py L624-L726](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L624-L726) [data\_loader\_utils\.py L264-L319](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L264-L319)

## Summary

 The data loading and parsing subsystem in RFAA provides a sophisticated framework for processing diverse biomolecular input formats and converting them into standardized feature representations\. This system handles:

 - Protein sequences and multiple sequence alignments
- Nucleic acid sequences
- Small molecule structures
- Template structures
- Multi\-chain complexes \(both heteromers and homomers\)

 The processed data feeds into the structure prediction pipeline, providing the neural network models with the features needed for accurate structure prediction\.

 Sources: [parsers\.py L1-L813](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/parsers.py#L1-L813) [data\_loader\_utils\.py L1-L910](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/data_loader_utils.py#L1-L910) [protein\.py L1-L94](https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/6c851405/rf2aa/data/protein.py#L1-L94)

---
*Source: [https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.3-data-loading-and-parsing](https://deepwiki.com/baker-laboratory/RoseTTAFold-All-Atom/5.3-data-loading-and-parsing) on DeepWiki*