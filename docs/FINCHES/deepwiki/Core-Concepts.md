# Core Concepts

> **Relevant source files**
> * [docs/extended_methods.rst](https://github.com/idptools/finches/blob/5b52ba40/docs/extended_methods.rst)
> * [finches/data/forcefield_dependencies.py](https://github.com/idptools/finches/blob/5b52ba40/finches/data/forcefield_dependencies.py)
> * [finches/epsilon_calculation.py](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py)

This document provides a deep dive into the fundamental concepts and algorithms underlying FINCHES calculations. It covers the core computational architecture, the central role of the `InteractionMatrixConstructor`, the forcefield interface pattern, matrix-based calculations, and sequence weighting mechanisms.

For detailed information about specific forcefield implementations, see [Forcefield Models](/idptools/finches/4.1-forcefield-models). For sequence processing specifics, see [Sequence Processing](/idptools/finches/4.2-sequence-processing). For matrix optimization details, see [Matrix Calculations](/idptools/finches/4.3-matrix-calculations).

## Computational Architecture Overview

FINCHES employs a layered architecture where user-facing frontend classes delegate to a central computational engine, which in turn utilizes pluggable forcefield models and optimized matrix operations.

### Core Architecture Mapping

```mermaid
flowchart TD

MF["Mpipi_frontend"]
CF["CALVADOS_frontend"]
FB["FinchesFrontend (base)"]
IMC["InteractionMatrixConstructor"]
EPS["epsilon_stateless.py functions"]
MM["matrix_manipulation.pyx"]
FF_IF["compute_interaction_parameter()"]
MPIPI["Mpipi_model"]
CALV["calvados_model"]
CUST["custom_model"]
PARSE["parsing_aminoacid_sequences"]
ST["sequence_tools"]
FD["FoldedDomain"]

MF --> IMC
CF --> IMC
IMC --> FF_IF
IMC --> PARSE
IMC --> FD

subgraph subGraph3 ["Processing Modules"]
    PARSE
    ST
    FD
    PARSE --> ST
end

subgraph subGraph2 ["Forcefield Interface"]
    FF_IF
    MPIPI
    CALV
    CUST
    FF_IF --> MPIPI
    FF_IF --> CALV
    FF_IF --> CUST
end

subgraph subGraph1 ["Computation Engine"]
    IMC
    EPS
    MM
    IMC --> EPS
    IMC --> MM
end

subgraph subGraph0 ["Frontend Layer"]
    MF
    CF
    FB
    FB --> MF
    FB --> CF
end
```

**Sources:** [finches/epsilon_calculation.py L18-L132](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L18-L132)

 [docs/extended_methods.rst L7-L10](https://github.com/idptools/finches/blob/5b52ba40/docs/extended_methods.rst#L7-L10)

## InteractionMatrixConstructor - The Central Engine

The `InteractionMatrixConstructor` class serves as the primary computational hub, implementing a standardized interface for calculating inter-residue interactions regardless of the underlying forcefield model.

### Core Responsibilities

The `InteractionMatrixConstructor` implements several key computational workflows:

| Method | Purpose | Returns |
| --- | --- | --- |
| `calculate_pairwise_heterotypic_matrix()` | Raw residue-residue interaction matrix | numpy array (len(seq1) × len(seq2)) |
| `calculate_weighted_pairwise_matrix()` | Matrix with charge/aliphatic weighting | numpy array with applied corrections |
| `calculate_epsilon_value()` | Single scalar interaction strength | float (attractive < 0, repulsive > 0) |
| `calculate_sliding_epsilon()` | Local interaction map via sliding windows | tuple (matrix, seq1_indices, seq2_indices) |

**Sources:** [finches/epsilon_calculation.py L396-L873](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L396-L873)

### Initialization and Dependencies

The constructor requires a forcefield parameters object and accepts several configuration options:

```mermaid
flowchart TD

PARAMS["parameters object"]
ALL_RES["ALL_RESIDUES_TYPES"]
COMPUTE["compute_interaction_parameter()"]
CONFIGS["CONFIGS dictionary"]
SEQ_CONV["sequence_converter function"]
CHARGE_PRE["charge_prefactor"]
NULL_BASE["null_interaction_baseline"]
LOOKUP["lookup[r1][r2] dictionary"]
VALID_GROUPS["valid_residue_groups"]

ALL_RES --> VALID_GROUPS
COMPUTE --> LOOKUP
CONFIGS --> CHARGE_PRE
CONFIGS --> NULL_BASE
SEQ_CONV --> VALID_GROUPS

subgraph subGraph2 ["Internal State"]
    LOOKUP
    VALID_GROUPS
end

subgraph subGraph1 ["Optional Configuration"]
    SEQ_CONV
    CHARGE_PRE
    NULL_BASE
end

subgraph subGraph0 ["Required Dependencies"]
    PARAMS
    ALL_RES
    COMPUTE
    CONFIGS
    PARAMS --> ALL_RES
    PARAMS --> COMPUTE
    PARAMS --> CONFIGS
end
```

**Sources:** [finches/epsilon_calculation.py L20-L191](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L20-L191)

 [finches/epsilon_calculation.py L195-L297](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L195-L297)

## Forcefield Interface Pattern

FINCHES uses an interface design pattern to decouple biophysical models from computational algorithms. This allows the same analysis code to work with different underlying force fields.

### Required Interface Methods

Every forcefield object must implement:

* `ALL_RESIDUES_TYPES`: List of valid residue groups (proteins, DNA, RNA)
* `compute_interaction_parameter(r1, r2)`: Returns interaction strength between residues
* `CONFIGS`: Dictionary containing `charge_prefactor` and `null_interaction_baseline`

**Sources:** [finches/epsilon_calculation.py L75-L90](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L75-L90)

### Lookup Table Construction

The `_update_lookup_dict()` method precomputes all pairwise interactions for performance:

```mermaid
flowchart TD

START["_update_lookup_dict()"]
RESET["Reset lookup = {}"]
EXTRACT["Extract valid_aa from ALL_RESIDUES_TYPES"]
LOOP1["For each r1 in valid_aa"]
LOOP2["For each r2 in valid_aa"]
COMPUTE["lookup[r1][r2] = compute_interaction_parameter(r1,r2)[0]"]
DONE["Lookup table ready"]

START --> RESET
RESET --> EXTRACT
EXTRACT --> LOOP1
LOOP1 --> LOOP2
LOOP2 --> COMPUTE
COMPUTE --> LOOP2
LOOP2 --> LOOP1
LOOP1 --> DONE
```

**Sources:** [finches/epsilon_calculation.py L195-L252](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L195-L252)

## Matrix-Based Epsilon Calculations

FINCHES calculates epsilon values through a multi-step matrix-based approach, separating raw interaction calculation from weighting and aggregation.

### Epsilon Calculation Pipeline

```mermaid
flowchart TD

SEQ["Input Sequences"]
CONV["sequence_converter()"]
RAW["calculate_pairwise_heterotypic_matrix()"]
WEIGHT["calculate_weighted_pairwise_matrix()"]
CHARGE["Charge weighting via get_charge_weighted_mask()"]
ALIPH["Aliphatic weighting via get_aliphatic_weighted_mask()"]
SPLIT["Split matrix at null_interaction_baseline"]
ATT["Attractive matrix (< baseline)"]
REP["Repulsive matrix (> baseline)"]
FINAL["Sum mean values"]

WEIGHT --> CHARGE
ALIPH --> SPLIT

subgraph subGraph2 ["Epsilon Derivation"]
    SPLIT
    ATT
    REP
    FINAL
    SPLIT --> ATT
    SPLIT --> REP
    ATT --> FINAL
    REP --> FINAL
end

subgraph subGraph1 ["Weighting Applications"]
    CHARGE
    ALIPH
    CHARGE --> ALIPH
end

subgraph subGraph0 ["Matrix Generation"]
    SEQ
    CONV
    RAW
    WEIGHT
    SEQ --> CONV
    CONV --> RAW
    RAW --> WEIGHT
end
```

**Sources:** [finches/epsilon_calculation.py L494-L603](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L494-L603)

 [docs/extended_methods.rst L13-L16](https://github.com/idptools/finches/blob/5b52ba40/docs/extended_methods.rst#L13-L16)

### Sliding Window Analysis

For interaction maps, FINCHES extracts subsquares from the full matrix and calculates local epsilon values:

* Window size must be odd (automatically corrected if even)
* Each subsquare treated as local `M^{raw-local}` matrix
* Uniform sampling with step size of 1
* Output indices account for window centering

**Sources:** [finches/epsilon_calculation.py L705-L873](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L705-L873)

 [docs/extended_methods.rst L18-L23](https://github.com/idptools/finches/blob/5b52ba40/docs/extended_methods.rst#L18-L23)

## Sequence Weighting Mechanisms

FINCHES applies two types of sequence-context weighting to capture local environmental effects.

### Charge Weighting Implementation

The charge correction downweighs repulsion between like-charged residues:

1. **Fragment Analysis**: For charged residue pairs, extract i±1 neighbors (4-6 residues total)
2. **Metrics Calculation**: Compute FCR (fraction charged) and NCPR (net charge per residue)
3. **Weighting Factor**: `|NCPR/FCR|` where same charges = 1, opposite charges = 0
4. **Scaling**: Apply forcefield-specific `charge_prefactor` (0-1 range)

The implementation reduces repulsive interactions: `w_matrix = matrix - (matrix * repulsive_mask * charge_prefactor)`

**Sources:** [finches/epsilon_calculation.py L546-L577](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L546-L577)

 [docs/extended_methods.rst L30-L38](https://github.com/idptools/finches/blob/5b52ba40/docs/extended_methods.rst#L30-L38)

### Aliphatic Weighting

Applied after charge weighting via `get_aliphatic_weighted_mask()` to enhance hydrophobic clustering effects.

**Sources:** [finches/epsilon_calculation.py L597-L599](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L597-L599)

## Performance Architecture

FINCHES employs strategic optimizations for computationally intensive operations.

### Cython Integration

```mermaid
flowchart TD

CALC_MAT["calculate_pairwise_heterotypic_matrix()"]
SLIDING["calculate_sliding_epsilon()"]
DICT2MAT["dict2matrix()"]
MAT_SCAN["matrix_scan()"]
SPEED["~8% of Python execution time"]

CALC_MAT --> DICT2MAT
SLIDING --> MAT_SCAN
DICT2MAT --> SPEED
MAT_SCAN --> SPEED

subgraph subGraph2 ["Performance Gains"]
    SPEED
end

subgraph subGraph1 ["Cython Acceleration"]
    DICT2MAT
    MAT_SCAN
end

subgraph subGraph0 ["Python Interface"]
    CALC_MAT
    SLIDING
end
```

The `use_cython=True` flag (default) enables optimized implementations in `matrix_manipulation.pyx` for matrix construction and sliding window analysis.

**Sources:** [finches/epsilon_calculation.py L440-L444](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L440-L444)

 [finches/epsilon_calculation.py L476-L477](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L476-L477)

 [finches/epsilon_calculation.py L831-L833](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L831-L833)

### Lookup Table Caching

All pairwise interactions are precomputed once during initialization and stored in `self.lookup[r1][r2]` for O(1) access during matrix construction.

**Sources:** [finches/epsilon_calculation.py L149](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L149-L149)

 [finches/epsilon_calculation.py L219-L237](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_calculation.py#L219-L237)