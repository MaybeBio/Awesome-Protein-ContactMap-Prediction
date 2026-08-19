# Structure Analysis Tools

> **Relevant source files**
> * [input_prep/parse_cif.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py)

This page covers the structure analysis and comparison utilities in RoseTTAFold2, focusing on TM-align integration and structure comparison tools. These utilities enable quantitative comparison of protein structures through TM-score calculations and structural alignments.

For information about PDBx/mmCIF parsing, see [PDBx/mmCIF Processing](/uw-ipd/RoseTTAFold2/7.1-pdbxmmcif-processing).

## Overview

The structure analysis tools provide functionality for:

* Integration with TM-align for structural comparisons
* Calculation of pairwise TM-scores between protein chains
* PDB file generation for structure visualization
* Structural alignment and transformation analysis

## TM-align Integration

The system integrates with the TM-align structural alignment tool to perform quantitative structure comparisons.

### TM-align Configuration

The TM-align executable path is configured as a global constant:

```
TMALIGN = '/home/aivan/prog/TMalign'
```

### TMalign Function

The `TMalign()` function provides the core interface for running TM-align comparisons between two protein chains.

```mermaid
flowchart TD

A["chainA"]
D["writepdb()"]
B["chainB"]
E["writepdb()"]
F["tempfile A"]
G["tempfile B"]
H["subprocess.Popen()"]
I["TMalign process"]
J["stdout parsing"]
K["transformation matrix"]
L["rmsd extraction"]
M["seqid extraction"]
N["TM-score extraction"]
O["rotation/translation"]
P["resAB result"]
Q["resBA result"]

F --> H
G --> H
J --> L
J --> M
J --> N
K --> O

subgraph subGraph2 ["Result Processing"]
    L
    M
    N
    O
    P
    Q
    L --> P
    M --> P
    N --> P
    O --> P
    P --> Q
end

subgraph subGraph1 ["TM-align Execution"]
    H
    I
    J
    K
    H --> I
    I --> J
    I --> K
end

subgraph subGraph0 ["Input Preparation"]
    A
    D
    B
    E
    F
    G
    A --> D
    B --> E
    D --> F
    E --> G
end
```

**TM-align Integration Workflow**

Sources: [input_prep/parse_cif.py L92-L152](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L92-L152)

The function performs the following steps:

1. **Temporary File Creation**: Creates temporary PDB files for both input chains
2. **Structure Writing**: Uses `writepdb()` to generate PDB format files
3. **TM-align Execution**: Runs TM-align as a subprocess with transformation output
4. **Result Parsing**: Extracts RMSD, sequence identity, TM-scores, and alignment data
5. **Transformation Processing**: Parses rotation and translation matrices

### Result Structure

The `TMalign()` function returns two dictionaries representing bidirectional alignments:

| Field | Description |
| --- | --- |
| `rmsd` | Root Mean Square Deviation |
| `seqid` | Sequence identity percentage |
| `tm` | TM-score for the alignment |
| `aln` | Alignment indices array |
| `t` | Translation vector |
| `u` | Rotation matrix |

Sources: [input_prep/parse_cif.py L149-L150](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L149-L150)

## Pairwise Structure Comparison

The `get_tm_pairs()` function computes TM-scores for all pairs of chains in a structure.

```mermaid
flowchart TD

A["chains dict"]
B["combinations()"]
C["chain pairs (A,B)"]
D["TMalign(chainA, chainB)"]
E["resAB, resBA"]
F["tm_pairs[(A,B)]"]
G["tm_pairs[(B,A)]"]
H["self chains"]
I["identity alignment"]
J["tm_pairs[(A,A)]"]
K["complete tm_pairs dict"]

C --> D
F --> K
G --> K
J --> K

subgraph subGraph2 ["Self-Alignment Addition"]
    H
    I
    J
    H --> I
    I --> J
end

subgraph subGraph1 ["Pairwise Alignment"]
    D
    E
    F
    G
    D --> E
    E --> F
    E --> G
end

subgraph subGraph0 ["Input Processing"]
    A
    B
    C
    A --> B
    B --> C
end
```

**Pairwise TM-Score Computation**

Sources: [input_prep/parse_cif.py L155-L173](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L155-L173)

### Self-Alignment Handling

For self-alignments, the system creates identity alignments with perfect scores:

```sql
tm_pairs.update({(A,A):{'rmsd':0.0, 'seqid':1.0, 'tm':1.0, 'aln':aln}})
```

Sources: [input_prep/parse_cif.py L167-L171](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L167-L171)

## PDB File Writing

The `writepdb()` function generates PDB format files from coordinate arrays and sequences.

### Amino Acid Mapping

The system maintains comprehensive amino acid and atom name mappings:

| Constant | Purpose |
| --- | --- |
| `RES_NAMES` | 20 standard amino acids (3-letter codes) |
| `RES_NAMES_1` | Single letter amino acid codes |
| `ATOM_NAMES` | Complete atom name definitions per residue |
| `idx2ra` | Index to residue-atom mapping |
| `aa2idx` | Amino acid-atom to index mapping |

Sources: [input_prep/parse_cif.py L16-L56](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L16-L56)

### PDB Writing Process

```mermaid
flowchart TD

A["xyz coordinates"]
D["coordinate iteration"]
B["sequence"]
C["bfac (optional)"]
E["residue-atom validation"]
F["NaN coordinate check"]
G["idx2ra mapping"]
H["PDB line formatting"]
I["PDB file writing"]
J["CA index tracking"]
K["index array return"]

D --> E
H --> I

subgraph subGraph2 ["Output Generation"]
    I
    J
    K
    I --> J
    J --> K
end

subgraph subGraph1 ["Atom Processing"]
    E
    F
    G
    H
    E --> F
    F --> G
    G --> H
end

subgraph subGraph0 ["Input Processing"]
    A
    D
    B
    C
    A --> D
    B --> D
    C --> D
end
```

**PDB File Writing Workflow**

Sources: [input_prep/parse_cif.py L58-L89](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L58-L89)

## Structure Analysis Workflow

The complete structure analysis workflow integrates mmCIF parsing with TM-align comparisons.

```mermaid
flowchart TD

A["mmCIF file"]
B["parse_mmcif()"]
C["chains dict"]
D["metadata dict"]
E["get_tm_pairs()"]
F["pairwise TMalign()"]
G["tm_pairs dict"]
H["TM-score matrix"]
I["metadata update"]
J["individual .pt files"]
K["metadata .pt file"]
L["chain structure"]
M["TM-score results"]
N["alignment data"]

C --> E
H --> I
C --> J
G --> L
G --> M
G --> N

subgraph subGraph3 ["Data Structures"]
    L
    M
    N
end

subgraph subGraph2 ["Output Generation"]
    I
    J
    K
    I --> K
end

subgraph subGraph1 ["TM-Score Analysis"]
    E
    F
    G
    H
    E --> F
    F --> G
    G --> H
end

subgraph subGraph0 ["Input Stage"]
    A
    B
    C
    D
    A --> B
    B --> C
    B --> D
end
```

**Complete Structure Analysis Pipeline**

Sources: [input_prep/parse_cif.py L444-L458](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L444-L458)

## Integration with RoseTTAFold2

The structure analysis tools integrate with the broader RoseTTAFold2 system by:

1. **Template Processing**: Providing structural similarity metrics for template selection
2. **Validation**: Enabling comparison of predicted structures with known structures
3. **Assembly Analysis**: Supporting multi-chain structure analysis through TM-score matrices
4. **Quality Assessment**: Providing quantitative measures for structure evaluation

The tools output structured data compatible with PyTorch tensors for seamless integration with the neural network pipeline.

Sources: [input_prep/parse_cif.py L466-L475](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/parse_cif.py#L466-L475)