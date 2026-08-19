# InputSpecification System

> **Relevant source files**
> * [models/rfd3/configs/inference_engine/rfdiffusion3.yaml](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/configs/inference_engine/rfdiffusion3.yaml)
> * [models/rfd3/configs/trainer/loss/losses/diffusion_loss.yaml](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/configs/trainer/loss/losses/diffusion_loss.yaml)
> * [models/rfd3/docs/.assets/input_selection_large.png](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/.assets/input_selection_large.png)
> * [models/rfd3/docs/examples/demo.json](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/examples/demo.json)
> * [models/rfd3/docs/input.md](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1)
> * [models/rfd3/src/rfd3/inference/input_parsing.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py)
> * [models/rfd3/src/rfd3/inference/legacy_input_parsing.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/legacy_input_parsing.py)
> * [models/rfd3/src/rfd3/model/RFD3_diffusion_module.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/model/RFD3_diffusion_module.py)
> * [models/rfd3/tests/test_bond_preservation_cases.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/tests/test_bond_preservation_cases.py)
> * [models/rfd3/tests/test_conditioning.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/tests/test_conditioning.py)
> * [models/rfd3/tests/test_legacy_ptm_bonds.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/tests/test_legacy_ptm_bonds.py)

## Purpose and Scope

The InputSpecification System is RFdiffusion3's declarative interface for defining protein design tasks. It provides a validated, type-safe specification language for describing what structures to generate, what constraints to apply, and how to condition the diffusion process. The system translates user-provided JSON/YAML configurations into annotated `AtomArray` objects ready for model inference.

This page covers the specification schema, validation pipeline, and the `InputSelection` mini-language. For information about running inference with these specifications, see [4.5](https://github.com/RosettaCommons/foundry/blob/cee116dc/4.5)

 For training-time usage, see [4.6](https://github.com/RosettaCommons/foundry/blob/cee116dc/4.6)

 For symmetry configuration details, see [4.3](https://github.com/RosettaCommons/foundry/blob/cee116dc/4.3)

---

## System Architecture

The InputSpecification System consists of three major components: specification parsing, validation, and atom array construction.

```mermaid
flowchart TD

JSON["JSON/YAML Files<br>(user-provided)"]
CLI["CLI Arguments<br>(Hydra overrides)"]
AtomArrayInput["Pre-loaded AtomArray<br>(programmatic)"]
ProcessInput["process_input()<br>engine.py"]
NormalizeInputs["normalize_inputs()<br>engine.py"]
EnsureAbsPath["ensure_input_is_abspath()<br>input_parsing.py"]
DesignInputSpec["DesignInputSpecification<br>input_parsing.py:113-231"]
InputSelection["InputSelection<br>parsing.py"]
SymmetryConfig["SymmetryConfig<br>symmetry_utils.py"]
SchemaValidation["validate_input_schema()<br>input_parsing.py:234-274"]
Canonicalize["canonicalize()<br>input_parsing.py:276-284"]
LoadInput["load_input()<br>input_parsing.py:286-362"]
AssignTypes["_assign_types_to_input()<br>input_parsing.py:417-498"]
AssertExclusivity["assert_exclusivity()<br>input_parsing.py:368-404"]
BuildInit["_build_init()<br>input_parsing.py:543-630"]
AccumulateComponents["accumulate_components()<br>input_parsing.py"]
AppendLigand["_append_ligand()<br>input_parsing.py:661-683"]
ApplySymmetry["_apply_symmetry()<br>input_parsing.py:685-694"]
SetOrigin["_set_origin()<br>input_parsing.py:696-720"]
ApplyGlobals["_apply_globals()<br>input_parsing.py:722-749"]
AnnotatedAtomArray["AtomArray<br>with conditioning annotations"]
Metadata["Specification Metadata<br>(for reproducibility)"]

JSON --> ProcessInput
CLI --> ProcessInput
AtomArrayInput --> DesignInputSpec
EnsureAbsPath --> DesignInputSpec
DesignInputSpec --> SchemaValidation
InputSelection --> LoadInput
SymmetryConfig --> ApplySymmetry
DesignInputSpec --> BuildInit
ApplyGlobals --> AnnotatedAtomArray
DesignInputSpec --> Metadata

subgraph Output ["Output"]
    AnnotatedAtomArray
    Metadata
end

subgraph subGraph4 ["Building Phase"]
    BuildInit
    AccumulateComponents
    AppendLigand
    ApplySymmetry
    SetOrigin
    ApplyGlobals
    BuildInit --> AccumulateComponents
    AccumulateComponents --> AppendLigand
    AppendLigand --> ApplySymmetry
    ApplySymmetry --> SetOrigin
    SetOrigin --> ApplyGlobals
end

subgraph subGraph3 ["Validation Pipeline"]
    SchemaValidation
    Canonicalize
    LoadInput
    AssignTypes
    AssertExclusivity
    SchemaValidation --> Canonicalize
    Canonicalize --> LoadInput
    LoadInput --> AssignTypes
    AssignTypes --> AssertExclusivity
end

subgraph subGraph2 ["Specification Core"]
    DesignInputSpec
    InputSelection
    SymmetryConfig
end

subgraph subGraph1 ["Parsing Layer"]
    ProcessInput
    NormalizeInputs
    EnsureAbsPath
    ProcessInput --> NormalizeInputs
    NormalizeInputs --> EnsureAbsPath
end

subgraph subGraph0 ["Input Sources"]
    JSON
    CLI
    AtomArrayInput
end
```

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L113-L777](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L113-L777)

 [models/rfd3/docs/input.md L1-L25](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L1-L25)

---

## Core Components

### DesignInputSpecification Class

The `DesignInputSpecification` is a Pydantic `BaseModel` that defines the complete schema for RFD3 input specifications [models/rfd3/src/rfd3/inference/input_parsing.py L113-L123](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L113-L123)

 It enforces strict validation at initialization time and provides methods for building annotated atom arrays.

```mermaid
flowchart TD

DataInputs["Data Inputs<br>input: str<br>atom_array_input: AtomArray<br>contig: InputSelection<br>unindex: InputSelection"]
Constraints["Constraints<br>length: str<br>ligand: str<br>cif_parser_args: dict"]
Conditioning["Conditioning Selections<br>select_fixed_atoms: InputSelection<br>select_unfixed_sequence: InputSelection<br>select_buried/exposed: InputSelection<br>select_hbond_donor/acceptor: InputSelection<br>select_hotspots: InputSelection"]
Globals["Global Conditioning<br>symmetry: SymmetryConfig<br>ori_token: list[float]<br>infer_ori_strategy: str<br>plddt_enhanced: bool<br>is_non_loopy: bool<br>partial_t: float"]
Meta["Metadata<br>extra: dict<br>dialect: int"]
PreValidation["Pre-Validation<br>validate_input_schema()<br>canonicalize()<br>load_input()"]
PostValidation["Post-Validation<br>assert_exclusivity()<br>attempt_expansion()<br>_assign_types_to_input()"]
Build["build()<br>→ AtomArray + metadata"]
BuildInit["_build_init()<br>→ initial AtomArray"]
Setters["_append_ligand()<br>_apply_symmetry()<br>_set_origin()<br>_apply_globals()"]

DataInputs --> PreValidation
Constraints --> PreValidation
Conditioning --> PreValidation
Globals --> PreValidation
Meta --> PreValidation
PostValidation --> Build

subgraph subGraph2 ["Build Methods"]
    Build
    BuildInit
    Setters
    Build --> BuildInit
    BuildInit --> Setters
end

subgraph subGraph1 ["Validators (model_validator)"]
    PreValidation
    PostValidation
    PreValidation --> PostValidation
end

subgraph subGraph0 ["DesignInputSpecification Fields"]
    DataInputs
    Constraints
    Conditioning
    Globals
    Meta
end
```

**Key Validation Rules:**

| Validator | Purpose | File Location |
| --- | --- | --- |
| `validate_input_schema` | Ensures either `input` or `contig`/`length` provided; validates usage consistency | [models/rfd3/src/rfd3/inference/input_parsing.py L234-L274](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L234-L274) |
| `canonicalize` | Normalizes `length` to string format, `input` to absolute path | [models/rfd3/src/rfd3/inference/input_parsing.py L276-L284](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L276-L284) |
| `load_input` | Loads atom array from file, initializes `InputSelection` objects | [models/rfd3/src/rfd3/inference/input_parsing.py L286-L362](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L286-L362) |
| `assert_exclusivity` | Validates mutually exclusive selections (RASA bins, motifs) | [models/rfd3/src/rfd3/inference/input_parsing.py L368-L404](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L368-L404) |
| `_assign_types_to_input` | Applies selection masks to atom array annotations | [models/rfd3/src/rfd3/inference/input_parsing.py L417-L498](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L417-L498) |

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L113-L231](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L113-L231)

 [models/rfd3/src/rfd3/inference/input_parsing.py L234-L404](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L234-L404)

---

### InputSelection Mini-Language

The `InputSelection` class provides a flexible mini-language for selecting atoms and residues within input structures [models/rfd3/src/rfd3/inference/parsing.py](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/parsing.py)

 It accepts three input formats: booleans, contig strings, or dictionaries with atom-level granularity.

```mermaid
flowchart TD

BoolInput["Boolean<br>true/false"]
StringInput["Contig String<br>'A1-10,B5-8'"]
DictInput["Dictionary<br>{'A1-2': 'BKBN',<br>'B5': 'N,CA,C'}"]
FromAny["from_any()<br>Factory method"]
ParseContig["Parse contig string"]
ParseDict["Parse dictionary"]
Storage["Internal Storage<br>dict[str, list[str]]"]
GetMask["get_mask()<br>→ np.ndarray[bool]"]
GetTokens["get_tokens()<br>→ dict[str, AtomArray]"]
GetKey["get(key)<br>→ list[str] | None"]
ShorthandValues["Shorthand Values<br>ALL → all atoms<br>BKBN → N,CA,C,O<br>TIP → residue-specific tip atom"]

BoolInput --> FromAny
StringInput --> FromAny
DictInput --> FromAny
Storage --> GetMask
Storage --> GetTokens
Storage --> GetKey
ShorthandValues --> ParseDict

subgraph subGraph3 ["Special Values"]
    ShorthandValues
end

subgraph subGraph2 ["Query Methods"]
    GetMask
    GetTokens
    GetKey
end

subgraph subGraph1 ["InputSelection Class"]
    FromAny
    ParseContig
    ParseDict
    Storage
    FromAny --> ParseContig
    FromAny --> ParseDict
    ParseContig --> Storage
    ParseDict --> Storage
end

subgraph subGraph0 ["Input Formats"]
    BoolInput
    StringInput
    DictInput
end
```

**Example Usage:**

```sql
# Boolean: select all/noneselect_fixed_atoms = InputSelection.from_any(True, atom_array) # Contig string: select residuesselect_fixed_atoms = InputSelection.from_any("A1-10,B5-8", atom_array) # Dictionary: atom-level controlselect_fixed_atoms = InputSelection.from_any({    "A1-2": "BKBN",      # N,CA,C,O for residues A1-2    "A3": "N,CA,C,O,CB", # Explicit atom list    "B5-7": "ALL",       # All atoms    "B10": "TIP"         # Residue-specific tip atom}, atom_array)
```

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L150-L165](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L150-L165)

 [models/rfd3/docs/input.md L95-L110](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L95-L110)

---

## Input File Format

The InputSpecification System accepts JSON or YAML files containing one or more design specifications. Each top-level key becomes an independent design task [models/rfd3/docs/input.md L34-L47](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L34-L47)

### JSON Structure

```mermaid
flowchart TD

Root["Root Object"]
GlobalArgs["global_args<br>(optional)<br>Applied to all specs"]
Spec1["'spec-1': {...}"]
Spec2["'spec-2': {...}"]
SpecN["'spec-N': {...}"]
Input["input: 'path/to/pdb'"]
Contig["contig: '50-80,/0,A1-100'"]
SelectUnfixed["select_unfixed_sequence: 'A20-35'"]
Ligand["ligand: 'HAX,OAA'"]
Symmetry["symmetry: {...}"]
Other["... other fields"]

Spec1 --> Input
Spec1 --> Contig
Spec1 --> SelectUnfixed
Spec1 --> Ligand
Spec1 --> Symmetry
Spec1 --> Other

subgraph subGraph1 ["Individual Specification"]
    Input
    Contig
    SelectUnfixed
    Ligand
    Symmetry
    Other
end

subgraph subGraph0 ["JSON/YAML File Structure"]
    Root
    GlobalArgs
    Spec1
    Spec2
    SpecN
    Root --> GlobalArgs
    Root --> Spec1
    Root --> Spec2
    Root --> SpecN
    GlobalArgs --> Spec1
    GlobalArgs --> Spec2
    GlobalArgs --> SpecN
end
```

**Example JSON:**

```json
{  "M0255_1mg5_unfixed": {    "input": "../input_pdbs/M0255_1mg5.pdb",     "ligand": "NAI,ACT",        "unindex": "A108,A139,A152,A156",    "length": "180-200",    "select_fixed_atoms": {      "A108": "ND2,CG",      "A139": "OG,CB,CA",      "A152": "OH,CZ",      "A156": "NZ,CE,CD",      "ACT": "OXT",      "NAI": ""    },    "allow_ligand_on_existing_chain": true  }}
```

**Sources:** [models/rfd3/docs/examples/demo.json L1-L16](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/examples/demo.json#L1-L16)

 [models/rfd3/docs/input.md L34-L47](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L34-L47)

---

## Processing Pipeline

The InputSpecification System processes inputs through a multi-stage pipeline that validates, loads, and constructs annotated atom arrays.

```mermaid
flowchart TD

LoadJSON["Load JSON/YAML"]
ApplyGlobalArgs["Apply global_args"]
SubsetKeys["Subset to keys"]
MakeAbsolute["Make paths absolute"]
SafeInit["DesignInputSpecification.safe_init()"]
CheckDialect["Check dialect field"]
LegacyPath["dialect=1<br>→ LegacySpecification"]
ModernPath["dialect=2<br>→ DesignInputSpecification"]
ValidateSchema["validate_input_schema()"]
Canonicalize["canonicalize()"]
LoadAtomArray["load_input()"]
CoerceSelections["Coerce to InputSelection"]
CheckExclusivity["assert_exclusivity()"]
AttemptExpansion["attempt_expansion()"]
AssignTypes["_assign_types_to_input()"]
BuildInit["_build_init()"]
GetTokens["Get indexed/unindexed tokens"]
SampleContig["Sample contig pattern"]
Accumulate["accumulate_components()"]
PostProcess["Post-process:<br>append ligand,<br>apply symmetry,<br>set origin,<br>apply globals"]
ValidateAnnotations["Validate annotations"]
ConvertBool["Convert to bool"]
ReturnArray["Return AtomArray + metadata"]

MakeAbsolute --> SafeInit
ModernPath --> ValidateSchema
CoerceSelections --> CheckExclusivity
AssignTypes --> BuildInit
PostProcess --> ValidateAnnotations

subgraph subGraph5 ["Stage 6: Output"]
    ValidateAnnotations
    ConvertBool
    ReturnArray
    ValidateAnnotations --> ConvertBool
    ConvertBool --> ReturnArray
end

subgraph subGraph4 ["Stage 5: Building"]
    BuildInit
    GetTokens
    SampleContig
    Accumulate
    PostProcess
    BuildInit --> GetTokens
    GetTokens --> SampleContig
    SampleContig --> Accumulate
    Accumulate --> PostProcess
end

subgraph subGraph3 ["Stage 4: Post-Validation"]
    CheckExclusivity
    AttemptExpansion
    AssignTypes
    CheckExclusivity --> AttemptExpansion
    AttemptExpansion --> AssignTypes
end

subgraph subGraph2 ["Stage 3: Pre-Validation"]
    ValidateSchema
    Canonicalize
    LoadAtomArray
    CoerceSelections
    ValidateSchema --> Canonicalize
    Canonicalize --> LoadAtomArray
    LoadAtomArray --> CoerceSelections
end

subgraph subGraph1 ["Stage 2: Specification Creation"]
    SafeInit
    CheckDialect
    LegacyPath
    ModernPath
    SafeInit --> CheckDialect
    CheckDialect --> LegacyPath
    CheckDialect --> ModernPath
end

subgraph subGraph0 ["Stage 1: File Loading"]
    LoadJSON
    ApplyGlobalArgs
    SubsetKeys
    MakeAbsolute
    LoadJSON --> ApplyGlobalArgs
    ApplyGlobalArgs --> SubsetKeys
    SubsetKeys --> MakeAbsolute
end
```

**Key Functions:**

| Function | Purpose | Location |
| --- | --- | --- |
| `DesignInputSpecification.safe_init()` | Factory method handling legacy/modern dialects | [models/rfd3/src/rfd3/inference/input_parsing.py L752-L762](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L752-L762) |
| `validate_input_schema()` | Validates field requirements and usage | [models/rfd3/src/rfd3/inference/input_parsing.py L234-L274](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L234-L274) |
| `load_input()` | Loads atom array and initializes selections | [models/rfd3/src/rfd3/inference/input_parsing.py L286-L362](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L286-L362) |
| `_assign_types_to_input()` | Applies selection masks to annotations | [models/rfd3/src/rfd3/inference/input_parsing.py L417-L498](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L417-L498) |
| `build()` | Orchestrates atom array construction | [models/rfd3/src/rfd3/inference/input_parsing.py L504-L537](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L504-L537) |

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L504-L777](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L504-L777)

---

## Conditioning Annotations

The InputSpecification System assigns conditioning annotations to atoms based on the provided selections [models/rfd3/src/rfd3/inference/input_parsing.py L417-L498](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L417-L498)

 These annotations control the diffusion process.

```mermaid
flowchart TD

FixedAtoms["select_fixed_atoms<br>→ is_motif_atom_with_fixed_coord"]
UnfixedSeq["select_unfixed_sequence<br>→ is_motif_atom_with_fixed_seq"]
Unindex["unindex<br>→ is_motif_atom_unindexed"]
Hotspots["select_hotspots<br>→ is_atom_level_hotspot"]
HbondDonor["select_hbond_donor<br>→ active_donor"]
HbondAcceptor["select_hbond_acceptor<br>→ active_acceptor"]
Buried["select_buried<br>→ rasa_bin=0"]
PartialBuried["select_partially_buried<br>→ rasa_bin=1"]
Exposed["select_exposed<br>→ rasa_bin=2"]
InitDefaults["Initialize defaults"]
PerTokenLoop["For each token (residue)"]
RedesignCheck["Check redesign_motif_sidechains"]
ApplyMask["Apply selection mask"]
CoordFixed["is_motif_atom_with_fixed_coord"]
SeqFixed["is_motif_atom_with_fixed_seq"]
Unindexed["is_motif_atom_unindexed"]
AtomHotspot["is_atom_level_hotspot"]
ActiveDonor["active_donor"]
ActiveAcceptor["active_acceptor"]
RASABin["rasa_bin"]

FixedAtoms --> InitDefaults
UnfixedSeq --> InitDefaults
Unindex --> InitDefaults
Hotspots --> InitDefaults
HbondDonor --> InitDefaults
HbondAcceptor --> InitDefaults
Buried --> InitDefaults
PartialBuried --> InitDefaults
Exposed --> InitDefaults
ApplyMask --> CoordFixed
ApplyMask --> SeqFixed
ApplyMask --> Unindexed
ApplyMask --> AtomHotspot
ApplyMask --> ActiveDonor
ApplyMask --> ActiveAcceptor
ApplyMask --> RASABin

subgraph subGraph2 ["Final Annotations"]
    CoordFixed
    SeqFixed
    Unindexed
    AtomHotspot
    ActiveDonor
    ActiveAcceptor
    RASABin
end

subgraph subGraph1 ["Application Logic"]
    InitDefaults
    PerTokenLoop
    RedesignCheck
    ApplyMask
    InitDefaults --> PerTokenLoop
    PerTokenLoop --> RedesignCheck
    RedesignCheck --> ApplyMask
end

subgraph subGraph0 ["Selection Fields"]
    FixedAtoms
    UnfixedSeq
    Unindex
    Hotspots
    HbondDonor
    HbondAcceptor
    Buried
    PartialBuried
    Exposed
end
```

**Required Annotations:**

| Annotation | Type | Purpose |
| --- | --- | --- |
| `is_motif_atom_with_fixed_coord` | `int` (bool) | Whether atom coordinates are fixed during diffusion |
| `is_motif_atom_with_fixed_seq` | `int` (bool) | Whether atom sequence identity is fixed |
| `is_motif_atom_unindexed` | `int` (bool) | Whether motif position is unknown to model |
| `is_atom_level_hotspot` | `int` (bool) | PPI hotspot conditioning |
| `active_donor` | `int` (bool) | Hydrogen bond donor atom |
| `active_acceptor` | `int` (bool) | Hydrogen bond acceptor atom |
| `rasa_bin` | `int` | Solvent accessibility: 0=buried, 1=partial, 2=exposed, 3=unconditioned |

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L417-L498](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L417-L498)

 [models/rfd3/src/rfd3/constants.py L30-L34](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/constants.py#L30-L34)

---

## Partial Diffusion Mode

When `partial_t` is specified, the InputSpecification System operates in partial diffusion mode [models/rfd3/docs/input.md L121-L147](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L121-L147)

```mermaid
flowchart TD

CheckPartialT["Check partial_t field"]
RequireInput["Require input file"]
DisallowLength["Disallow length argument"]
SkipSampling["Skip contig sampling"]
AssignTypes["assign_types_()"]
SubsetProtein["Subset to protein"]
AppendUnindexed["Append unindexed components"]
CheckSymmetry["Check if symmetry present"]
SkipCOMCenter["Skip COM centering"]
UseCOM["Use COM for origin"]
KeepCoords["Keep all coordinates"]
SetPartialT["Set partial_t annotation"]

DisallowLength --> SkipSampling
AppendUnindexed --> CheckSymmetry
KeepCoords --> SetPartialT

subgraph subGraph3 ["Partial T Annotation"]
    SetPartialT
end

subgraph subGraph2 ["Modified Origin Setting"]
    CheckSymmetry
    SkipCOMCenter
    UseCOM
    KeepCoords
    CheckSymmetry --> SkipCOMCenter
    CheckSymmetry --> UseCOM
    SkipCOMCenter --> KeepCoords
    UseCOM --> KeepCoords
end

subgraph subGraph1 ["Modified Building"]
    SkipSampling
    AssignTypes
    SubsetProtein
    AppendUnindexed
    SkipSampling --> AssignTypes
    AssignTypes --> SubsetProtein
    SubsetProtein --> AppendUnindexed
end

subgraph subGraph0 ["Partial Diffusion Detection"]
    CheckPartialT
    RequireInput
    DisallowLength
    CheckPartialT --> RequireInput
    RequireInput --> DisallowLength
end
```

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L195-L198](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L195-L198)

 [models/rfd3/docs/input.md L121-L147](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/docs/input.md?plain=1#L121-L147)

---

## Legacy Compatibility

The system maintains backwards compatibility with original input formats through the `LegacySpecification` class [models/rfd3/src/rfd3/inference/input_parsing.py L78-L105](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L78-L105)

```mermaid
flowchart TD

CheckDialect["Check 'dialect' field"]
Dialect1["dialect=1<br>(Legacy)"]
Dialect2["dialect=2<br>(Modern)"]
LegacySpec["LegacySpecification"]
LegacyBuild["create_atom_array_from_design_specification_legacy()"]
ModernSpec["DesignInputSpecification"]
ModernBuild["build()"]

Dialect1 --> LegacySpec
Dialect2 --> ModernSpec

subgraph subGraph2 ["Modern Path"]
    ModernSpec
    ModernBuild
    ModernSpec --> ModernBuild
end

subgraph subGraph1 ["Legacy Path"]
    LegacySpec
    LegacyBuild
    LegacySpec --> LegacyBuild
end

subgraph subGraph0 ["Dialect Selection"]
    CheckDialect
    Dialect1
    Dialect2
    CheckDialect --> Dialect1
    CheckDialect --> Dialect2
end
```

**Migration Guide:**

| Legacy Field | Modern Field | Notes |
| --- | --- | --- |
| `unfix_sequence` | `select_unfixed_sequence` | Now uses `InputSelection` |
| `contig_atoms` | `select_fixed_atoms` | Now uses `InputSelection` |
| `atomwise_rasa` | `select_buried`/`select_exposed` | Split into explicit fields |

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L78-L105](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L78-L105)

 [models/rfd3/src/rfd3/inference/legacy_input_parsing.py L416-L611](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/legacy_input_parsing.py#L416-L611)

---

## Field Reference

### Complete Field Table

| Field | Type | Description |
| --- | --- | --- |
| `input` | `str` | Path to input PDB/CIF file [models/rfd3/src/rfd3/inference/input_parsing.py L130](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L130-L130) |
| `contig` | `InputSelection` | Contig specification string [models/rfd3/src/rfd3/inference/input_parsing.py L132](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L132-L132) |
| `unindex` | `InputSelection` | Unindexed components selection [models/rfd3/src/rfd3/inference/input_parsing.py L133](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L133-L133) |
| `length` | `str` | Length range as 'min-max' or int [models/rfd3/src/rfd3/inference/input_parsing.py L139](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L139-L139) |
| `ligand` | `str` | Ligand name or index to include [models/rfd3/src/rfd3/inference/input_parsing.py L140](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L140-L140) |
| `select_fixed_atoms` | `InputSelection` | Atoms to fix coordinates for [models/rfd3/src/rfd3/inference/input_parsing.py L150](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L150-L150) |
| `select_unfixed_sequence` | `InputSelection` | Components to unfix sequence for [models/rfd3/src/rfd3/inference/input_parsing.py L158](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L158-L158) |
| `select_buried` | `InputSelection` | Selection of RASA buried conditioning [models/rfd3/src/rfd3/inference/input_parsing.py L168](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L168-L168) |
| `select_hotspots` | `InputSelection` | Selection of hotspot residues [models/rfd3/src/rfd3/inference/input_parsing.py L182](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L182-L182) |
| `partial_t` | `float` | Partial diffusion noise level [models/rfd3/src/rfd3/inference/input_parsing.py L198](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L198-L198) |
| `dialect` | `int` | RFdiffusion3 input dialect (1: legacy, 2: release) [models/rfd3/src/rfd3/inference/input_parsing.py L144](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L144-L144) |

**Sources:** [models/rfd3/src/rfd3/inference/input_parsing.py L121-L229](https://github.com/RosettaCommons/foundry/blob/cee116dc/models/rfd3/src/rfd3/inference/input_parsing.py#L121-L229)