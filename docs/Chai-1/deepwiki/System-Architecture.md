# System Architecture

> **Relevant source files**
> * [chai_lab/chai1.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py)
> * [chai_lab/data/collate/collate.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/collate/collate.py)
> * [chai_lab/data/dataset/all_atom_feature_context.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py)
> * [chai_lab/data/dataset/msas/utils.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py)

This document provides a comprehensive overview of the Chai-1 system architecture, covering the core inference pipeline, data processing components, and model structure. For specific implementation details about input processing, see [Input Processing](/chaidiscovery/chai-lab/4-input-processing). For feature generation systems, see [Feature Generation](/chaidiscovery/chai-lab/5-feature-generation).

## Overview

Chai-1 is a multi-modal foundation model for molecular structure prediction that unifies prediction of proteins, small molecules, DNA, RNA, and glycosylations. The system follows a pipeline architecture with distinct stages for input processing, feature generation, model inference, and output generation.

## High-Level System Architecture

### Core Pipeline Components

```mermaid
flowchart TD

CLI["chai-lab CLI"]
WebUI["Web Server<br>lab.chaidiscovery.com"]
PythonAPI["Python API<br>run_inference()"]
FASTAParser["FASTA Parser<br>read_inputs()"]
ChainLoader["Chain Loader<br>load_chains_from_raw()"]
Validator["Input Validator<br>raise_if_too_many_tokens()"]
MSAGen["MSA Generator<br>generate_colabfold_msas()"]
TemplateLoader["Template Loader<br>get_template_context()"]
ESMEmbedder["ESM Embedder<br>get_esm_embedding_context()"]
FeatureFactory["FeatureFactory<br>feature_generators"]
FeatureEmbed["Feature Embedding<br>feature_embedding.pt"]
TokenEmbed["Token Embedder<br>token_embedder.pt"]
Trunk["Trunk Module<br>trunk.pt"]
Diffusion["Diffusion Module<br>diffusion_module.pt"]
Confidence["Confidence Head<br>confidence_head.pt"]
Ranking["Structure Ranking<br>rank()"]
CIFOutput["CIF Output<br>save_to_cif()"]
ScoreCalc["Score Calculation<br>get_scores()"]

CLI --> FASTAParser
WebUI --> FASTAParser
PythonAPI --> FASTAParser
Validator --> MSAGen
Validator --> TemplateLoader
Validator --> ESMEmbedder
FeatureFactory --> FeatureEmbed
Confidence --> Ranking

subgraph subGraph4 ["Output Processing"]
    Ranking
    CIFOutput
    ScoreCalc
    Ranking --> CIFOutput
    Ranking --> ScoreCalc
end

subgraph subGraph3 ["Core Inference Engine"]
    FeatureEmbed
    TokenEmbed
    Trunk
    Diffusion
    Confidence
    FeatureEmbed --> TokenEmbed
    TokenEmbed --> Trunk
    Trunk --> Diffusion
    Diffusion --> Confidence
end

subgraph subGraph2 ["Feature Generation"]
    MSAGen
    TemplateLoader
    ESMEmbedder
    FeatureFactory
    MSAGen --> FeatureFactory
    TemplateLoader --> FeatureFactory
    ESMEmbedder --> FeatureFactory
end

subgraph subGraph1 ["Input Processing"]
    FASTAParser
    ChainLoader
    Validator
    FASTAParser --> ChainLoader
    ChainLoader --> Validator
end

subgraph subGraph0 ["User Interface Layer"]
    CLI
    WebUI
    PythonAPI
end
```

Sources: [chai_lab/chai1.py L499-L1059](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L499-L1059)

 [README.md L23-L46](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/README.md?plain=1#L23-L46)

### Data Context Architecture

The `AllAtomFeatureContext` serves as the central hub for all molecular and feature data required for inference. It encapsulates various sub-contexts that are eventually collated into tensors for the model.

```mermaid
flowchart TD

AllAtomFC["AllAtomFeatureContext<br>Unified feature container"]
AllAtomSC["AllAtomStructureContext<br>Molecular structure data"]
MSAContext["MSAContext<br>Multiple sequence alignments"]
TemplateContext["TemplateContext<br>Structural templates"]
EmbeddingContext["EmbeddingContext<br>ESM embeddings"]
RestraintContext["RestraintContext<br>User constraints"]
FASTAInput["FASTA File<br>Sequences"]
MSAFiles["MSA Files<br>.aligned.pqt"]
TemplateFiles["Template Files<br>.m8 hits"]
RestraintFiles["Restraint Files<br>User constraints"]
Collate["Collate<br>Batch preparation"]
FeatureGen["Feature Generators<br>feature_generators dict"]

FASTAInput --> AllAtomSC
MSAFiles --> MSAContext
TemplateFiles --> TemplateContext
RestraintFiles --> RestraintContext
AllAtomFC --> Collate

subgraph subGraph2 ["Feature Processing"]
    Collate
    FeatureGen
    Collate --> FeatureGen
end

subgraph subGraph1 ["Input Sources"]
    FASTAInput
    MSAFiles
    TemplateFiles
    RestraintFiles
end

subgraph subGraph0 ["Core Data Structures"]
    AllAtomFC
    AllAtomSC
    MSAContext
    TemplateContext
    EmbeddingContext
    RestraintContext
    AllAtomSC --> AllAtomFC
    MSAContext --> AllAtomFC
    TemplateContext --> AllAtomFC
    EmbeddingContext --> AllAtomFC
    RestraintContext --> AllAtomFC
end
```

Sources: [chai_lab/data/dataset/all_atom_feature_context.py L25-L40](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L25-L40)

 [chai_lab/data/collate/collate.py L24-L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/collate/collate.py#L24-L35)

 [chai_lab/chai1.py L486-L495](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L486-L495)

## Model Inference Pipeline

### Trunk Recycling and Diffusion Process

The inference engine utilizes a sequential execution flow involving feature embedding, trunk recycling, and a diffusion denoising loop. The `ModuleWrapper` handles JIT-loaded components like `trunk.pt` and `diffusion_module.pt`.

```mermaid
flowchart TD

ComponentCache["_component_cache<br>ModuleWrapper storage"]
ComponentLoader["load_exported()<br>JIT module loader"]
FeatureEmbed["feature_embedding.pt<br>Embed input features"]
TokenEmbed["token_embedder.pt<br>Process token representations"]
BondEmbed["bond_loss_input_proj.pt<br>Bond feature processing"]
TrunkInit["Initial Representations<br>token_single_initial_repr"]
TrunkRecycle["Trunk Recycling<br>num_trunk_recycles=3"]
MSASubsample["MSA Subsampling<br>subsample_and_reorder_msa_feats_n_mask()"]
TrunkModule["trunk.pt<br>Core processing module"]
NoiseSchedule["InferenceNoiseSchedule<br>Diffusion timesteps"]
DiffusionLoop["Diffusion Loop<br>num_diffn_timesteps=200"]
DiffusionModule["diffusion_module.pt<br>Structure denoising"]
AtomPositions["Atom Positions<br>3D coordinates"]
ConfidenceHead["confidence_head.pt<br>Quality prediction"]
PAELogits["PAE Logits<br>Predicted aligned error"]
PDELogits["PDE Logits<br>Predicted distance error"]
PLDDTLogits["pLDDT Logits<br>Local confidence"]

ComponentCache --> FeatureEmbed
BondEmbed --> TrunkInit
TrunkModule --> NoiseSchedule
AtomPositions --> ConfidenceHead

subgraph subGraph4 ["Confidence Prediction"]
    ConfidenceHead
    PAELogits
    PDELogits
    PLDDTLogits
    ConfidenceHead --> PAELogits
    ConfidenceHead --> PDELogits
    ConfidenceHead --> PLDDTLogits
end

subgraph subGraph3 ["Diffusion Denoising"]
    NoiseSchedule
    DiffusionLoop
    DiffusionModule
    AtomPositions
    NoiseSchedule --> DiffusionLoop
    DiffusionLoop --> DiffusionModule
    DiffusionModule --> AtomPositions
end

subgraph subGraph2 ["Trunk Processing"]
    TrunkInit
    TrunkRecycle
    MSASubsample
    TrunkModule
    TrunkInit --> TrunkRecycle
    TrunkRecycle --> MSASubsample
    MSASubsample --> TrunkModule
    TrunkModule --> TrunkRecycle
end

subgraph subGraph1 ["Feature Embedding"]
    FeatureEmbed
    TokenEmbed
    BondEmbed
    FeatureEmbed --> TokenEmbed
    TokenEmbed --> BondEmbed
end

subgraph subGraph0 ["Model Loading"]
    ComponentCache
    ComponentLoader
    ComponentLoader --> ComponentCache
end
```

Sources: [chai_lab/chai1.py L115-L137](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L115-L137)

 [chai_lab/chai1.py L139-L149](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L139-L149)

 [chai_lab/chai1.py L744-L778](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L744-L778)

 [chai_lab/chai1.py L821-L886](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L821-L886)

 [chai_lab/chai1.py L894-L915](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L894-L915)

 [chai_lab/data/dataset/msas/utils.py L51-L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/utils.py#L51-L86)

## Feature Generation System

### Feature Factory Architecture

The system uses a `FeatureFactory` that orchestrates multiple `feature_generators` to produce tensors for the model. These generators cover sequence, structure, MSA, and restraint-based features.

| Feature Type | Generator Class | Purpose |
| --- | --- | --- |
| Sequence Features | `RelativeSequenceSeparation` | Positional relationships [chai_lab/chai1.py L173](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L173-L173) |
| Token Features | `RelativeTokenSeparation` | Token-level distances [chai_lab/chai1.py L174](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L174-L174) |
| Structure Features | `BlockedAtomPairDistogram` | Atomic distance features [chai_lab/chai1.py L179](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L179-L179) |
| MSA Features | `MSAProfileGenerator` | Sequence alignment profiles [chai_lab/chai1.py L72](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L72-L72) |
| Template Features | `TemplateDistogramGenerator` | Structural template features [chai_lab/chai1.py L86](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L86-L86) |
| Restraint Features | `TokenDistanceRestraint` | User-defined constraints [chai_lab/chai1.py L93](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L93-L93) |

```mermaid
flowchart TD

SeqGen["Sequence Generators<br>RelativeSequenceSeparation<br>RelativeTokenSeparation"]
StructGen["Structure Generators<br>BlockedAtomPairDistogram<br>AtomElementOneHot"]
MSAGen["MSA Generators<br>MSAProfileGenerator<br>MSAFeatureGenerator"]
TemplateGen["Template Generators<br>TemplateDistogramGenerator<br>TemplateMaskGenerator"]
RestraintGen["Restraint Generators<br>TokenDistanceRestraint<br>TokenPairPocketRestraint"]
FeatureFactory["FeatureFactory<br>feature_generators dict"]
FeatureType["FeatureType<br>TOKEN, ATOM, ATOM_PAIR, etc."]

SeqGen --> FeatureFactory
StructGen --> FeatureFactory
MSAGen --> FeatureFactory
TemplateGen --> FeatureFactory
RestraintGen --> FeatureFactory

subgraph subGraph1 ["Feature Factory"]
    FeatureFactory
    FeatureType
    FeatureFactory --> FeatureType
end

subgraph subGraph0 ["Feature Generators"]
    SeqGen
    StructGen
    MSAGen
    TemplateGen
    RestraintGen
end
```

Sources: [chai_lab/chai1.py L172-L236](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L172-L236)

 [chai_lab/data/features/feature_factory.py L13-L25](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/feature_factory.py#L13-L25)

## External Service Integration

### ColabFold MSA Generation

MSA generation is integrated via ColabFold, with support for template discovery and processing into the `TemplateContext`.

```mermaid
flowchart TD

ProteinSeqs["Protein Sequences<br>EntityType.PROTEIN"]
ColabFoldAPI["ColabFold API<br>msa_server_url"]
A3MFiles["A3M Files<br>Raw alignments"]
AlignedPQT["Aligned PQT Files<br>Processed alignments"]
MSAContexts["MSA Contexts<br>get_msa_contexts()"]
TemplateHits["Template Hits<br>.m8 format"]
RCSBDownload["RCSB Download<br>Structure files"]
TemplateAlign["Template Alignment<br>Sequence to structure"]
TemplateCtx["TemplateContext<br>Processed templates"]

ColabFoldAPI --> TemplateHits

subgraph subGraph1 ["Template Processing"]
    TemplateHits
    RCSBDownload
    TemplateAlign
    TemplateCtx
    TemplateHits --> RCSBDownload
    RCSBDownload --> TemplateAlign
    TemplateAlign --> TemplateCtx
end

subgraph subGraph0 ["MSA Processing"]
    ProteinSeqs
    ColabFoldAPI
    A3MFiles
    AlignedPQT
    MSAContexts
    ProteinSeqs --> ColabFoldAPI
    ColabFoldAPI --> A3MFiles
    A3MFiles --> AlignedPQT
    AlignedPQT --> MSAContexts
end
```

Sources: [chai_lab/chai1.py L389-L402](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L389-L402)

 [chai_lab/chai1.py L422-L438](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L422-L438)

 [chai_lab/data/dataset/msas/colabfold.py L20-L45](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/msas/colabfold.py#L20-L45)

## Configuration and Validation

### System Limits and Validation

The system enforces several limits to ensure computational feasibility, defined in the feature context and model metadata.

| Limit Type | Maximum Value | Source |
| --- | --- | --- |
| Token Count | `max(AVAILABLE_MODEL_SIZES)` | [chai_lab/data/collate/utils.py L22](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/collate/utils.py#L22-L22) |
| Template Count | `MAX_NUM_TEMPLATES = 4` | [chai_lab/data/dataset/all_atom_feature_context.py L21](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L21-L21) |
| MSA Depth | `MAX_MSA_DEPTH = 16384` | [chai_lab/data/dataset/all_atom_feature_context.py L20](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L20-L20) |

Sources: [chai_lab/chai1.py L255-L277](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L255-L277)

 [chai_lab/data/dataset/all_atom_feature_context.py L20-L21](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/all_atom_feature_context.py#L20-L21)

## Output Generation and Ranking

### Structure Candidate Processing

The final stage involves ranking predicted structures based on confidence metrics and saving them to standard formats like CIF.

```mermaid
flowchart TD

DiffusionOutput["Diffusion Output<br>atom_pos coordinates"]
ConfidenceScores["Confidence Scores<br>PAE, PDE, pLDDT"]
NumSamples["Sample Count<br>num_diffn_samples"]
FrameCalc["Frame Calculation<br>get_frames_and_mask()"]
RankFunc["Ranking Function<br>rank()"]
SampleRanking["SampleRanking<br>Aggregate scores"]
CIFFiles["CIF Files<br>save_to_cif()"]
ScoreFiles["Score Files<br>scores.model_idx_*.npz"]
MSAPlot["MSA Plot<br>plot_msa()"]
StructureCandidates["StructureCandidates<br>Sorted by score"]

DiffusionOutput --> FrameCalc
ConfidenceScores --> RankFunc
SampleRanking --> CIFFiles
SampleRanking --> ScoreFiles
SampleRanking --> StructureCandidates
MSAPlot --> StructureCandidates

subgraph subGraph3 ["Final Structure"]
    StructureCandidates
end

subgraph subGraph2 ["Output Files"]
    CIFFiles
    ScoreFiles
    MSAPlot
end

subgraph subGraph1 ["Ranking System"]
    FrameCalc
    RankFunc
    SampleRanking
    FrameCalc --> RankFunc
    RankFunc --> SampleRanking
end

subgraph subGraph0 ["Structure Generation"]
    DiffusionOutput
    ConfidenceScores
    NumSamples
end
```

Sources: [chai_lab/chai1.py L989-L1051](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L989-L1051)

 [chai_lab/ranking/rank.py L104-L105](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/ranking/rank.py#L104-L105)

 [chai_lab/data/io/cif_utils.py L98](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/io/cif_utils.py#L98-L98)

## Memory Management and Device Handling

The system implements sophisticated memory management for handling large models via a JIT module cache and transient device movement.

### Component Caching and Device Movement

```mermaid
flowchart TD

ComponentCache["_component_cache<br>dict[str, ModuleWrapper]"]
ComponentMoved["_component_moved_to()<br>Context manager"]
DeviceMovement["Device Movement<br>CPU ↔ GPU transfers"]
JITLoad["torch.jit.load<br>Model deserialization"]
ModuleWrapper["ModuleWrapper<br>Forward method wrapper"]
LowMemory["low_memory flag<br>CPU offloading"]
FeatureEmbed["feature_embedding.pt"]
TokenEmbed["token_embedder.pt"]
Trunk["trunk.pt"]
Diffusion["diffusion_module.pt"]
Confidence["confidence_head.pt"]

ModuleWrapper --> ComponentCache
ComponentMoved --> FeatureEmbed
ComponentMoved --> TokenEmbed
ComponentMoved --> Trunk
ComponentMoved --> Diffusion
ComponentMoved --> Confidence
LowMemory --> DeviceMovement

subgraph subGraph2 ["Inference Components"]
    FeatureEmbed
    TokenEmbed
    Trunk
    Diffusion
    Confidence
end

subgraph subGraph1 ["Model Loading"]
    JITLoad
    ModuleWrapper
    LowMemory
    JITLoad --> ModuleWrapper
end

subgraph subGraph0 ["Memory Management"]
    ComponentCache
    ComponentMoved
    DeviceMovement
    ComponentCache --> ComponentMoved
    ComponentMoved --> DeviceMovement
end
```

Sources: [chai_lab/chai1.py L151-L167](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L151-L167)

 [chai_lab/chai1.py L115-L148](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/chai1.py#L115-L148)