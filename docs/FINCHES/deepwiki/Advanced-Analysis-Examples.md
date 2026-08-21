# Advanced Analysis Examples

> **Relevant source files**
> * [demo/nucleic_acid/protein_nucleic_acid_binding.ipynb](https://github.com/idptools/finches/blob/5b52ba40/demo/nucleic_acid/protein_nucleic_acid_binding.ipynb)
> * [demo/phase_diagrams/phase_diagram_demo.ipynb](https://github.com/idptools/finches/blob/5b52ba40/demo/phase_diagrams/phase_diagram_demo.ipynb)
> * [demo/protein_matrix/interaction_matrix_demo.ipynb](https://github.com/idptools/finches/blob/5b52ba40/demo/protein_matrix/interaction_matrix_demo.ipynb)
> * [finches/epsilon_to_FHtheory.py](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_to_FHtheory.py)

This page demonstrates complex analytical workflows using FINCHES, including phase diagram generation with Flory-Huggins theory, protein-nucleic acid interaction analysis, and multi-condition parameter sweeps. For basic usage patterns such as simple epsilon calculations and interaction maps, see [Basic Usage Examples](/idptools/finches/6.1-basic-usage-examples).

## Phase Diagram Analysis with Flory-Huggins Theory

FINCHES integrates analytical solutions to the Flory-Huggins model for generating phase diagrams from computed epsilon values. This functionality bridges molecular-level interaction parameters with thermodynamic phase behavior.

### Core Phase Diagram Workflow

```mermaid
flowchart TD

SEQ["Protein Sequence"]
EPSILON["epsilon_stateless.get_sequence_epsilon_value()"]
CONVERT["epsilon_to_FHtheory.epsilon_to_phase_diagram()"]
BINODAL["floryhuggins.calculate_binodal()"]
SPINODAL["floryhuggins.calculate_spinodal()"]
PHASE_DATA["Phase Separation Boundaries"]
PLOT["Phase Diagram Visualization"]
EPSILON_CALC["Negative epsilon = attractive"]
LENGTH_NORM["epsilon/sequence_length"]
CHI_CONV["chi = delta_eps / (kB * T)"]

SEQ --> EPSILON
EPSILON --> CONVERT
CONVERT --> BINODAL
CONVERT --> SPINODAL
BINODAL --> PHASE_DATA
SPINODAL --> PHASE_DATA
PHASE_DATA --> PLOT
CONVERT --> EPSILON_CALC

subgraph subGraph0 ["Conversion Logic"]
    EPSILON_CALC
    LENGTH_NORM
    CHI_CONV
    EPSILON_CALC --> LENGTH_NORM
    LENGTH_NORM --> CHI_CONV
end
```

The phase diagram calculation involves several key transformations. When epsilon values are positive (repulsive), they are set to a small attractive value (-0.01) to prevent mathematical issues in the Flory-Huggins calculations. The epsilon value is normalized by sequence length since it represents chain-level interactions while the Flory-Huggins model requires per-residue energies.

**Sources:** [finches/epsilon_to_FHtheory.py L356-L448](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_to_FHtheory.py#L356-L448)

### Multi-Condition Phase Analysis

FINCHES provides specialized functions for analyzing phase behavior across different environmental conditions:

```mermaid
flowchart TD

INPUT["Sequence + Frontend"]
SALT_SWEEP["build_SALT_dependent_phase_diagrams()"]
PH_SWEEP["build_PH_dependent_phase_diagrams()"]
DIELEC_SWEEP["build_DIELECTRIC_dependent_phase_diagrams()"]
SALT_LOOP["Loop: condition_list values"]
PH_LOOP["Loop: condition_list values"]
DIELEC_LOOP["Loop: condition_list values"]
UPDATE_PARAMS["X_class.parameters.salt = value"]
UPDATE_PARAMS2["X_class.parameters.pH = value"]
UPDATE_PARAMS3["X_class.parameters.dielectric = value"]
LOOKUP_UPDATE["X_class._update_lookup_dict()"]
PHASE_CALC["return_phase_diagram()"]
RESULTS["Condition-indexed results"]

INPUT --> SALT_SWEEP
INPUT --> PH_SWEEP
INPUT --> DIELEC_SWEEP
SALT_SWEEP --> SALT_LOOP
PH_SWEEP --> PH_LOOP
DIELEC_SWEEP --> DIELEC_LOOP
SALT_LOOP --> UPDATE_PARAMS
PH_LOOP --> UPDATE_PARAMS2
DIELEC_LOOP --> UPDATE_PARAMS3
UPDATE_PARAMS --> LOOKUP_UPDATE
UPDATE_PARAMS2 --> LOOKUP_UPDATE
UPDATE_PARAMS3 --> LOOKUP_UPDATE
LOOKUP_UPDATE --> PHASE_CALC
PHASE_CALC --> RESULTS
```

Each condition sweep function follows the same pattern: iterating through condition values, updating the forcefield parameters, refreshing the lookup tables, and computing phase diagrams for each condition.

**Sources:** [finches/epsilon_to_FHtheory.py L33-L271](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_to_FHtheory.py#L33-L271)

## Protein-Nucleic Acid Interaction Analysis

FINCHES supports heterotypic interactions between proteins and nucleic acids, extending beyond traditional protein-protein interactions. This capability is particularly relevant for analyzing RNA-binding proteins and DNA-binding domains.

### Nucleic Acid Parameter Integration

```mermaid
flowchart TD

PROTEIN_SEQ["Protein Sequence"]
MPIPI["Mpipi_frontend"]
RNA_SEQ["RNA/DNA Sequence"]
CUSTOM_PARAMS["Custom Nucleic Acid Parameters"]
MATRIX["InteractionMatrixConstructor"]
EPSILON["Cross-species epsilon calculation"]
INTERMAP["Heterotypic interaction maps"]
MPIPI_PARAMS["Mpipi amino acid parameters"]
NA_PARAMS["Nucleic acid interaction parameters"]

PROTEIN_SEQ --> MPIPI
RNA_SEQ --> CUSTOM_PARAMS
MPIPI --> MATRIX
CUSTOM_PARAMS --> MATRIX
MATRIX --> EPSILON
MATRIX --> INTERMAP
MPIPI_PARAMS --> MATRIX
NA_PARAMS --> MATRIX

subgraph subGraph0 ["Parameter Sources"]
    MPIPI_PARAMS
    NA_PARAMS
end
```

The analysis involves extending the standard amino acid parameter sets to include nucleic acid residues, enabling calculation of protein-nucleic acid interaction energies using the same computational framework.

**Sources:** [demo/nucleic_acid/protein_nucleic_acid_binding.ipynb L1-L200](https://github.com/idptools/finches/blob/5b52ba40/demo/nucleic_acid/protein_nucleic_acid_binding.ipynb#L1-L200)

### Heterotypic Interaction Matrices

When analyzing protein-nucleic acid systems, interaction matrices reveal binding specificity patterns:

```mermaid
flowchart TD

PROT_RES["Protein Residues (rows)"]
MATRIX["Interaction Matrix"]
NA_RES["Nucleic Acid Residues (cols)"]
HOTSPOTS["Binding Hotspots"]
SPECIFICITY["Sequence Specificity"]
AVIDITY["Multivalent Interactions"]
SPATIAL["Spatial interaction patterns"]
ENERGETIC["Energy landscape mapping"]
COMPETITIVE["Competitive binding analysis"]

PROT_RES --> MATRIX
NA_RES --> MATRIX
MATRIX --> HOTSPOTS
MATRIX --> SPECIFICITY
MATRIX --> AVIDITY
HOTSPOTS --> SPATIAL
SPECIFICITY --> ENERGETIC
AVIDITY --> COMPETITIVE

subgraph subGraph0 ["Analysis Types"]
    SPATIAL
    ENERGETIC
    COMPETITIVE
end
```

**Sources:** [demo/nucleic_acid/protein_nucleic_acid_binding.ipynb L1-L200](https://github.com/idptools/finches/blob/5b52ba40/demo/nucleic_acid/protein_nucleic_acid_binding.ipynb#L1-L200)

## Complex Interaction Matrix Analysis

Advanced interaction matrix analysis goes beyond simple pairwise epsilon calculations to examine spatially-resolved interaction patterns and multi-domain systems.

### IDR-IDR Interaction Networks

```mermaid
flowchart TD

IDR1["med1_idr sequence"]
FRONTEND["Mpipi_frontend/CALVADOS_frontend"]
IDR2["med14_idr sequence"]
IDR3["med15_idr sequence"]
MATRIX_CALC["generate_interaction_matrix()"]
HOMO1["med1-med1 homotypic"]
HOMO2["med14-med14 homotypic"]
HOMO3["med15-med15 homotypic"]
HETERO12["med1-med14 heterotypic"]
HETERO13["med1-med15 heterotypic"]
HETERO23["med14-med15 heterotypic"]
HEATMAPS["Interaction heatmaps"]
VECTORS["Per-residue interaction vectors"]
WINDOWS["Sliding window analysis"]

IDR1 --> FRONTEND
IDR2 --> FRONTEND
IDR3 --> FRONTEND
FRONTEND --> MATRIX_CALC
MATRIX_CALC --> HOMO1
MATRIX_CALC --> HOMO2
MATRIX_CALC --> HOMO3
MATRIX_CALC --> HETERO12
MATRIX_CALC --> HETERO13
MATRIX_CALC --> HETERO23
HOMO1 --> HEATMAPS
HETERO12 --> HEATMAPS
HOMO1 --> VECTORS
HETERO12 --> VECTORS
HOMO1 --> WINDOWS
HETERO12 --> WINDOWS

subgraph subGraph0 ["Analysis Outputs"]
    HEATMAPS
    VECTORS
    WINDOWS
end
```

The interaction matrix analysis reveals position-specific interaction preferences across different IDR pairs, enabling identification of binding motifs and cooperative interaction regions.

**Sources:** [demo/protein_matrix/interaction_matrix_demo.ipynb L80-L150](https://github.com/idptools/finches/blob/5b52ba40/demo/protein_matrix/interaction_matrix_demo.ipynb#L80-L150)

### Multi-Scale Analysis Pipeline

```mermaid
flowchart TD

SEQUENCE["Input Sequences"]
GLOBAL["Global epsilon calculation"]
LOCAL["Local interaction matrices"]
DOMAINS["Domain-specific analysis"]
PHASE["Phase behavior prediction"]
BINDING["Binding site identification"]
MODULARITY["Modular interaction analysis"]
CHAIN["Chain-level (epsilon)"]
RESIDUE["Residue-level (matrices)"]
SEGMENT["Segment-level (domains)"]
COMBINED["Multi-scale interaction model"]

SEQUENCE --> GLOBAL
SEQUENCE --> LOCAL
SEQUENCE --> DOMAINS
GLOBAL --> PHASE
LOCAL --> BINDING
DOMAINS --> MODULARITY
PHASE --> CHAIN
BINDING --> RESIDUE
MODULARITY --> SEGMENT
CHAIN --> COMBINED
RESIDUE --> COMBINED
SEGMENT --> COMBINED

subgraph Integration ["Integration"]
    COMBINED
end

subgraph subGraph0 ["Resolution Levels"]
    CHAIN
    RESIDUE
    SEGMENT
end
```

**Sources:** [demo/protein_matrix/interaction_matrix_demo.ipynb L1-L200](https://github.com/idptools/finches/blob/5b52ba40/demo/protein_matrix/interaction_matrix_demo.ipynb#L1-L200)

 [finches/epsilon_to_FHtheory.py L275-L351](https://github.com/idptools/finches/blob/5b52ba40/finches/epsilon_to_FHtheory.py#L275-L351)

## Custom Forcefield Integration

Advanced users can integrate custom interaction parameters for specialized systems or novel amino acid modifications.

### Custom Parameter Workflow

```mermaid
flowchart TD

CUSTOM_DICT["Custom interaction dictionary"]
CUSTOM_MODEL["custom_model.CustomModel"]
LITERATURE["Literature parameters"]
EXPERIMENT["Experimental data"]
CONSTRUCTOR["InteractionMatrixConstructor"]
STANDARD_FF["Mpipi/CALVADOS"]
HYBRID["Hybrid parameter calculations"]
RESULTS["Analysis with custom parameters"]
SIGMA["Size parameters (sigma)"]
EPSILON_PARAM["Energy parameters (epsilon)"]
CHARGE["Charge modifications"]
CUSTOM_PROPS["Custom properties"]

CUSTOM_DICT --> CUSTOM_MODEL
LITERATURE --> CUSTOM_DICT
EXPERIMENT --> CUSTOM_DICT
CUSTOM_MODEL --> CONSTRUCTOR
STANDARD_FF --> CONSTRUCTOR
CONSTRUCTOR --> HYBRID
HYBRID --> RESULTS
CUSTOM_DICT --> SIGMA
CUSTOM_DICT --> EPSILON_PARAM
CUSTOM_DICT --> CHARGE
CUSTOM_DICT --> CUSTOM_PROPS

subgraph subGraph0 ["Parameter Types"]
    SIGMA
    EPSILON_PARAM
    CHARGE
    CUSTOM_PROPS
end
```

Custom forcefields enable analysis of non-standard systems including modified amino acids, synthetic polymers, or hybrid protein-polymer systems.

**Sources:** [finches/forcefields/custom_model.py L1-L100](https://github.com/idptools/finches/blob/5b52ba40/finches/forcefields/custom_model.py#L1-L100)

## Integration with External Tools

FINCHES analysis workflows can be integrated with structure prediction and experimental validation tools.

### Analysis Integration Pipeline

```mermaid
flowchart TD

METAPREDICT["metapredict disorder prediction"]
SEQUENCES["IDR identification"]
SPARROW["sparrow structure analysis"]
FINCHES["FINCHES interaction analysis"]
PHASE_PRED["Phase separation prediction"]
BINDING_PRED["Binding specificity prediction"]
VALIDATION["Experimental validation"]
STRUCT_PRED["Structure prediction tools"]
EXP_DATA["Experimental datasets"]
ML_MODELS["Machine learning models"]

METAPREDICT --> SEQUENCES
SPARROW --> SEQUENCES
SEQUENCES --> FINCHES
FINCHES --> PHASE_PRED
FINCHES --> BINDING_PRED
PHASE_PRED --> VALIDATION
BINDING_PRED --> VALIDATION
VALIDATION --> STRUCT_PRED
VALIDATION --> EXP_DATA
VALIDATION --> ML_MODELS

subgraph subGraph0 ["External Integration"]
    STRUCT_PRED
    EXP_DATA
    ML_MODELS
end
```

**Sources:** [demo/protein_matrix/interaction_matrix_demo.ipynb L55-L65](https://github.com/idptools/finches/blob/5b52ba40/demo/protein_matrix/interaction_matrix_demo.ipynb#L55-L65)

 [demo/nucleic_acid/protein_nucleic_acid_binding.ipynb L55-L65](https://github.com/idptools/finches/blob/5b52ba40/demo/nucleic_acid/protein_nucleic_acid_binding.ipynb#L55-L65)