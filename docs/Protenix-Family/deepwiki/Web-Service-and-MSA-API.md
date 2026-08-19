# Web Service and MSA API

> **Relevant source files**
> * [protenix/web_service/colab_request_parser.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py)
> * [protenix/web_service/colab_request_utils.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py)
> * [protenix/web_service/prediction_visualization.py](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py)
> * [runner/msa_search.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/msa_search.py)
> * [scripts/colabfold_msa.py](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/colabfold_msa.py)

## Purpose and Scope

This document covers the web service infrastructure for Protenix, specifically the `RequestParser` system that handles web-based prediction requests and integrates with the MMseqs2 API for Multiple Sequence Alignment (MSA) generation. This system provides a higher-level interface than the CLI for running predictions through web services.

For information about the CLI-based MSA generation tools, see [Multiple Sequence Alignment](/bytedance/Protenix/3.3-multiple-sequence-alignment). For details on how MSA features are embedded in the model, see [Neural Network Components](/bytedance/Protenix/5.2-neural-network-components). For batch inference execution, see [Running Inference](/bytedance/Protenix/3.4-running-inference).

---

## System Architecture

The web service layer bridges external prediction requests with the core inference pipeline. It handles data preparation, MSA generation via external services, and orchestration of the inference process.

### High-Level Request Flow

```mermaid
flowchart TD

RequestJSON["Request JSON<br>(sequences, bonds, constraints)"]
Email["Email<br>(for MSA service)"]
Parser["RequestParser<br>colab_request_parser.py"]
DataCache["download_data_cache()<br>components.cif, clusters"]
ModelCache["get_model()<br>checkpoint download"]
JSONGen["get_data_json()<br>input JSON creation"]
MSASearch["msa_search()<br>run_mmseqs2_service()"]
MMseqs2["MMseqs2 Service<br>MMSEQS_SERVICE_HOST_URL"]
MSAPost["msa_postprocess()<br>pairing/non-pairing split"]
TemplateSearch["run_template_search()<br>hmmsearch.a3m"]
InferenceRunner["runner/inference.py<br>subprocess call"]
Output["CIF structures + confidence"]

RequestJSON --> Parser
Email --> Parser
JSONGen --> MSASearch
MSAPost --> TemplateSearch
MSAPost --> InferenceRunner
TemplateSearch --> InferenceRunner
DataCache --> InferenceRunner
ModelCache --> InferenceRunner

subgraph subGraph4 ["Inference Execution"]
    InferenceRunner
    Output
    InferenceRunner --> Output
end

subgraph subGraph3 ["Template Search"]
    TemplateSearch
end

subgraph subGraph2 ["MSA Service Integration"]
    MSASearch
    MMseqs2
    MSAPost
    MSASearch --> MMseqs2
    MMseqs2 --> MSAPost
end

subgraph subGraph1 ["RequestParser System"]
    Parser
    DataCache
    ModelCache
    JSONGen
    Parser --> DataCache
    Parser --> ModelCache
    Parser --> JSONGen
end

subgraph subGraph0 ["External Input"]
    RequestJSON
    Email
end
```

**Sources:** [protenix/web_service/colab_request_parser.py L86-L497](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L86-L497)

---

## RequestParser Class

The `RequestParser` class is the main orchestrator for web-based prediction requests. It handles the complete workflow from receiving a request JSON to launching the inference process.

### Initialization and Configuration

```mermaid
flowchart TD

Init["RequestParser.init()"]
RequestJSON["request_json_path"]
RequestDir["request_dir<br>(working directory)"]
Email["email<br>(optional)"]
ModelName["model_name<br>(default: protenix_base_default_v1.0.0)"]
LoadJSON["Load request JSON"]
MakeDir["Create request_dir"]

RequestJSON --> Init
RequestDir --> Init
Email --> Init
ModelName --> Init
Init --> LoadJSON
Init --> MakeDir
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `request_json_path` | str | Required | Path to input request JSON file |
| `request_dir` | str | Required | Working directory for outputs and intermediate files |
| `email` | str | `""` | Email for MMseqs2 service (optional) |
| `model_name` | str | `"protenix_base_default_v1.0.0"` | Model variant to use for inference |

**Sources:** [protenix/web_service/colab_request_parser.py L86-L100](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L86-L100)

### Request JSON Format

The request JSON specifies the input complex, prediction parameters, and whether to use MSA/template features.

| Field | Type | Description |
| --- | --- | --- |
| `name` | str | Name identifier for the complex |
| `sequences` | List[Dict] | Sequence specifications (see below) |
| `covalent_bonds` | List | Bond specifications between entities |
| `use_msa` | bool | Whether to generate protein MSA |
| `use_template` | bool | Whether to search for templates |
| `atom_confidence` | bool | Whether to compute atom-level confidence |
| `model_seeds` | List[int] | Random seeds for model runs (optional) |
| `N_sample` | int | Number of diffusion samples (optional) |
| `N_step` | int | Number of diffusion steps (optional) |
| `N_cycle` | int | Number of recycling iterations (optional) |

Each sequence entry is a dictionary with a single key indicating the molecule type:

| Sequence Type | Fields | Description |
| --- | --- | --- |
| `proteinChain` | `sequence`: str | Protein amino acid sequence |
| `dnaSequence` | `sequence`: str | DNA nucleotide sequence |
| `rnaSequence` | `sequence`: str | RNA nucleotide sequence |
| `ligand` | `ccdCode` or `smiles` | Ligand specification |
| `ion` | `ccdCode` | Ion specification |

**Sources:** [protenix/web_service/colab_request_parser.py L137-L166](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L137-L166)

---

## Data and Model Management

### Data Cache Download

The `download_data_cache()` method fetches required reference data files from a remote server if not already present locally.

```mermaid
flowchart TD

Start["download_data_cache()"]
CacheDir["DATA_CACHE_DIR<br>(PROTENIX_ROOT_DIR/common/)"]
CheckCCD["Check components.cif"]
CheckRDKit["Check components.cif.rdkit_mol.pkl"]
CheckCluster["Check clusters-by-entity-40.txt"]
DownloadCCD["download_tos_url()<br>URL['ccd_components_file']"]
DownloadRDKit["download_tos_url()<br>URL['ccd_components_rdkit_mol_file']"]
DownloadCluster["download_tos_url()<br>URL['pdb_cluster_file']"]
SaveCCD["Save to cache_dir"]
SaveRDKit["Save to cache_dir"]
SaveCluster["Save to cache_dir"]
Return["Return cache_paths dict"]

Start --> CheckCCD
Start --> CheckRDKit
Start --> CheckCluster
CheckCCD --> DownloadCCD
CheckRDKit --> DownloadRDKit
CheckCluster --> DownloadCluster
DownloadCCD --> SaveCCD
DownloadRDKit --> SaveRDKit
DownloadCluster --> SaveCluster
SaveCCD --> Return
SaveRDKit --> Return
SaveCluster --> Return
```

The cache files are stored in `$PROTENIX_ROOT_DIR/common/`:

| File | Purpose | Size Constraint |
| --- | --- | --- |
| `components.cif` | CCD chemical component definitions | Required for ligand/ion parsing |
| `components.cif.rdkit_mol.pkl` | Pre-computed RDKit molecules | Speeds up ligand processing |
| `clusters-by-entity-40.txt` | PDB entity clustering at 40% identity | Used for template filtering |

**Sources:** [protenix/web_service/colab_request_parser.py L102-L118](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L102-L118)

### Model Checkpoint Download

The `get_model()` method ensures the requested model checkpoint is available locally.

```mermaid
flowchart TD

GetModel["get_model()"]
CheckpointDir["CHECKPOINT_DIR<br>(PROTENIX_ROOT_DIR/checkpoint/)"]
CheckExists["Check if<br>{model_name}.pt exists"]
Download["download_model()<br>download_tos_url()"]
Return["Return checkpoint_dir"]
SaveCheckpoint["Save to<br>checkpoint_dir/{model_name}.pt"]
Verify["Verify file exists"]
Error["Raise ValueError"]

GetModel --> CheckExists
CheckExists --> Download
CheckExists --> Return
Download --> SaveCheckpoint
SaveCheckpoint --> Verify
Verify --> Return
Verify --> Error
```

Model checkpoints are stored in `$PROTENIX_ROOT_DIR/checkpoint/{model_name}.pt`. The download URL is retrieved from the `URL` dictionary keyed by model name.

**Sources:** [protenix/web_service/colab_request_parser.py L120-L135](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L120-L135)

---

## MSA Service Integration

The system integrates with the MMseqs2 service for protein MSA generation through a web API. This provides a faster alternative to running local MSA searches.

### Service Configuration

```
MMSEQS_SERVICE_HOST_URL = os.getenv(    "MMSEQS_SERVICE_HOST_URL", "https://protenix-server.com/api/msa")
```

The service URL can be overridden via the `MMSEQS_SERVICE_HOST_URL` environment variable.

**Sources:** [protenix/web_service/colab_request_parser.py L38-L40](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L38-L40)

### MSA Search Modes

The system supports two operational modes:

| Mode | Endpoint | Use Case | Output Files |
| --- | --- | --- | --- |
| `protenix` | `/ticket/msa` | Default Protenix workflow | `{idx}.a3m`, `uniref_tax.m8`, `pdb70_220313_db.m8` |
| `colabfold` | `/ticket/msa` or `/ticket/pair` | ColabFold compatibility | `bfd.mgnify30.metaeuk30.smag30.a3m`, `uniref.a3m`, `pair.a3m` |

**Sources:** [protenix/web_service/colab_request_parser.py L243-L331](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L243-L331)

### Request Submission and Polling

```mermaid
sequenceDiagram
  participant RequestParser
  participant MMseqs2 Service
  participant Result Storage

  RequestParser->>MMseqs2 Service: POST /ticket/msa
  MMseqs2 Service-->>RequestParser: (query sequences, mode, email)
  loop [Status Polling (every 60s)]
    RequestParser->>MMseqs2 Service: {status: "PENDING", id: "job_id"}
    MMseqs2 Service-->>RequestParser: GET /ticket/{job_id}
  end
  RequestParser->>MMseqs2 Service: {status: "RUNNING"}
  MMseqs2 Service-->>RequestParser: GET /ticket/{job_id}
  RequestParser->>MMseqs2 Service: {status: "COMPLETE"}
  MMseqs2 Service-->>Result Storage: GET /result/download/{job_id}
  Result Storage-->>RequestParser: out.tar.gz
  RequestParser->>RequestParser: MSA files
```

The submission and polling logic is implemented in the `run_mmseqs2_service()` function:

1. **Submit Job**: POST to `/ticket/msa` with query sequences [protenix/web_service/colab_request_utils.py L80-L86](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L80-L86)
2. **Poll Status**: GET `/ticket/{job_id}` every 60 seconds [protenix/web_service/colab_request_utils.py L113-L118](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L113-L118)
3. **Handle States**: * `PENDING`/`RUNNING`: Continue polling [protenix/web_service/colab_request_utils.py L228-L232](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L228-L232) * `COMPLETE`: Proceed to download [protenix/web_service/colab_request_utils.py L233-L234](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L233-L234) * `ERROR`/`MAINTENANCE`: Raise exception [protenix/web_service/colab_request_utils.py L235-L241](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L235-L241) * `RATELIMIT`/`UNKNOWN`: Retry after delay [protenix/web_service/colab_request_utils.py L242-L254](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L242-L254)
4. **Download Results**: GET `/result/download/{job_id}` [protenix/web_service/colab_request_utils.py L146-L151](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L146-L151)
5. **Extract**: Unpack tar.gz to working directory [protenix/web_service/colab_request_utils.py L259-L262](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L259-L262)

**Sources:** [protenix/web_service/colab_request_utils.py L44-L262](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L44-L262)

### Error Handling and Retries

```mermaid
flowchart TD

Submit["submit(seqs, mode, N)"]
Try["Try POST request<br>(timeout=6.02s)"]
CheckStatus["Check response status"]
Retry["Log warning, retry"]
CountError["Increment error_count"]
CheckCount["error_count > 5?"]
Raise["Raise Exception"]
Sleep["Sleep 5s, retry"]
Status["Status Type"]
SleepRetry["Sleep 60s<br>Resubmit"]
RaiseError["Raise Exception:<br>Invalid input or server error"]
RaiseMaint["Raise Exception:<br>Maintenance mode"]
Return["Return job ID"]

Submit --> Try
Try --> CheckStatus
Try --> Retry
Try --> CountError
Retry --> Try
CountError --> CheckCount
CheckCount --> Raise
CheckCount --> Sleep
Sleep --> Try
CheckStatus --> Status
Status --> SleepRetry
Status --> RaiseError
Status --> RaiseMaint
Status --> Return
SleepRetry --> Submit
```

The system implements robust retry logic:

* **Connection Timeouts**: Set to 6.02s (slightly > 3 × 2s for TCP retries) [protenix/web_service/colab_request_utils.py L79-L83](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L79-L83)
* **Error Retry Limit**: Maximum 5 retries before raising exception [protenix/web_service/colab_request_utils.py L97-L98](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L97-L98)
* **Rate Limiting**: 60s sleep when encountering `RATELIMIT` status [protenix/web_service/colab_request_utils.py L242-L244](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L242-L244)
* **Unknown States**: 60s sleep and resubmit for `UNKNOWN` status [protenix/web_service/colab_request_utils.py L249-L254](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L249-L254)

**Sources:** [protenix/web_service/colab_request_utils.py L69-L140](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L69-L140)

 [protenix/web_service/colab_request_utils.py L225-L257](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_utils.py#L225-L257)

---

## MSA Processing Pipeline

After downloading raw MSA files from the service, they must be post-processed into the format expected by Protenix.

### MSA File Processing Workflow

```mermaid
flowchart TD

RawMSA["Raw MSA Files<br>{idx}.a3m, uniref_tax.m8"]
ReadM8["read_m8()<br>Parse uniref_to_ncbi_taxid mapping"]
ReadA3M["read_a3m()<br>Parse headers and sequences"]
TaxMap["UniRef ID → NCBI Tax ID"]
MSAData["Headers + Sequences + uniref_index"]
Process["make_pairing_and_non_pairing_msa()"]
IterateSeqs["Iterate MSA sequences"]
CheckQuery["Sequence == query?"]
Skip["Skip (don't duplicate query)"]
CheckIndex["idx < uniref_index/2?"]
AddTaxID["Add TaxID to header<br>e.g., UniRef100_{id}_{taxid}/"]
KeepHeader["Keep header as-is"]
Pairing["Append to pairing.a3m"]
NonPairing["Append to non_pairing.a3m"]
Output["Output Files:<br>pairing.a3m<br>non_pairing.a3m"]

RawMSA --> ReadM8
RawMSA --> ReadA3M
ReadM8 --> TaxMap
ReadA3M --> MSAData
TaxMap --> Process
MSAData --> Process
Process --> IterateSeqs
IterateSeqs --> CheckQuery
CheckQuery --> Skip
CheckQuery --> CheckIndex
CheckIndex --> AddTaxID
CheckIndex --> KeepHeader
AddTaxID --> Pairing
KeepHeader --> NonPairing
Pairing --> Output
NonPairing --> Output
```

**Sources:** [protenix/web_service/colab_request_parser.py L333-L461](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L333-L461)

### Pairing vs Non-Pairing MSA

The system splits raw MSA into two categories:

| MSA Type | Source | Purpose | Format |
| --- | --- | --- | --- |
| `pairing.a3m` | UniRef100 hits from first half of MSA | Paired MSA for multi-chain complexes | Headers include NCBI taxonomy ID [protenix/web_service/colab_request_parser.py L374-L381](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L374-L381) |
| `non_pairing.a3m` | ColabfoldDB and remaining UniRef hits | General sequence homology | Standard headers without taxonomy [protenix/web_service/colab_request_parser.py L387-L396](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L387-L396) |

The split is determined by the `uniref_index` marker in the raw MSA file, which indicates where UniRef sequences end and ColabfoldDB sequences begin [protenix/web_service/colab_request_parser.py L363-L365](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L363-L365)

**Sources:** [protenix/web_service/colab_request_parser.py L363-L396](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L363-L396)

### Fallback: Dummy MSA

If MSA search fails or returns no results, the system generates a dummy MSA containing only the query sequence:

```mermaid
flowchart TD

CheckMSA["MSA files exist?"]
MakeDummy["make_dummy_msa()"]
Process["Process MSA"]
WriteQuery[">query<br>{sequence}"]
BothFiles["Write to both<br>pairing.a3m and non_pairing.a3m"]
CheckM8["uniref_tax.m8 exists?"]
Pairing["Split pairing/non-pairing"]
NonPairingOnly["non_pairing.a3m only"]
DummyPairing["Create dummy pairing.a3m"]

CheckMSA --> MakeDummy
CheckMSA --> Process
MakeDummy --> WriteQuery
WriteQuery --> BothFiles
Process --> CheckM8
CheckM8 --> Pairing
CheckM8 --> NonPairingOnly
NonPairingOnly --> DummyPairing
```

**Sources:** [protenix/web_service/colab_request_parser.py L413-L428](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L413-L428)

 [protenix/web_service/colab_request_parser.py L454-L458](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L454-L458)

---

## Template Search Integration

When `use_template=true` in the request, the system performs template search after MSA generation:

```mermaid
flowchart TD

MSAFiles["pairing.a3m<br>non_pairing.a3m"]
UseTemplate["use_template?"]
RunSearch["run_template_search()<br>runner/template_search.py"]
Skip["Skip template search"]
HMMSearch["hmmsearch against<br>pdb_seqres database"]
TemplateFile["hmmsearch.a3m"]
AddPath["Add templatesPath<br>to sequence entry"]

MSAFiles --> UseTemplate
UseTemplate --> RunSearch
UseTemplate --> Skip
RunSearch --> HMMSearch
HMMSearch --> TemplateFile
TemplateFile --> AddPath
```

**Sources:** [protenix/web_service/colab_request_parser.py L224-L236](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L224-L236)

---

## Input JSON Generation

The `get_data_json()` method orchestrates the complete workflow from request JSON to inference-ready input JSON:

### Data JSON Generation Workflow

```mermaid
flowchart TD

Start["get_data_json()"]
ParseSeqs["Parse request sequences"]
IdentifyMSA["Identify entities<br>needing MSA"]
TempJSON["Create temporary JSON<br>(for size validation)"]
Features["SampleDictToFeatures"]
GetAtoms["get_atom_array()"]
CheckAtoms["num_atoms > 60000?"]
ErrorAtoms["Raise TooLargeComplexError"]
CheckTokens["num_tokens > 5000?"]
ErrorTokens["Raise TooLargeComplexError"]
ProceedMSA["Entities need MSA?"]
MSASearch["msa_search()<br>run_mmseqs2_service()"]
BuildJSON["Build final input JSON"]
MSAPost["msa_postprocess()"]
MapPaths["Map MSA paths to entities"]
TemplateCheck["use_template?"]
Template["run_template_search()"]
WriteJSON["Write inputs.json"]
Return["Return input_json_path"]

Start --> ParseSeqs
ParseSeqs --> IdentifyMSA
IdentifyMSA --> TempJSON
TempJSON --> Features
Features --> GetAtoms
GetAtoms --> CheckAtoms
CheckAtoms --> ErrorAtoms
CheckAtoms --> CheckTokens
CheckTokens --> ErrorTokens
CheckTokens --> ProceedMSA
ProceedMSA --> MSASearch
ProceedMSA --> BuildJSON
MSASearch --> MSAPost
MSAPost --> MapPaths
MapPaths --> TemplateCheck
TemplateCheck --> Template
TemplateCheck --> BuildJSON
Template --> BuildJSON
BuildJSON --> WriteJSON
WriteJSON --> Return
```

**Sources:** [protenix/web_service/colab_request_parser.py L137-L241](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L137-L241)

### Size Validation

The system enforces limits to prevent out-of-memory errors:

```
MAX_ATOM_NUM = 60000MAX_TOKEN_NUM = 5000
```

If the complex exceeds these limits, a `TooLargeComplexError` is raised [protenix/web_service/colab_request_parser.py L179-L182](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L179-L182)

**Sources:** [protenix/web_service/colab_request_parser.py L41-L83](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L41-L83)

 [protenix/web_service/colab_request_parser.py L176-L182](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L176-L182)

---

## Inference Launch

The `launch()` method constructs and executes the inference command:

```mermaid
flowchart TD

Launch["launch()"]
GetJSON["get_data_json()<br>Generate input JSON + MSA"]
GetModel["get_model()<br>Ensure checkpoint available"]
BuildCmd["Build command string"]
BaseCmd["python3 runner/inference.py<br>--load_checkpoint_dir<br>--model_name<br>--dump_dir<br>--input_json_path<br>--need_atom_confidence<br>--use_msa<br>--num_workers 0"]
AddSeeds["model_seeds<br>in request?"]
AppendSeeds["--seeds \seed1,seed2,...\"]
AddSampling["N_sample/N_step<br>in request?"]
AppendSampling["--sample_diffusion.N_sample<br>--sample_diffusion.N_step"]
AddCycle["N_cycle<br>in request?"]
AppendCycle["--model.N_cycle"]
AddTemplate["use_template<br>in request?"]
AppendTemplate["--use_template true"]
Execute["subprocess.call(command, shell=True)"]

Launch --> GetJSON
Launch --> GetModel
GetJSON --> BuildCmd
GetModel --> BuildCmd
BuildCmd --> BaseCmd
BaseCmd --> AddSeeds
AddSeeds --> AppendSeeds
AddSeeds --> AddSampling
AppendSeeds --> AddSampling
AddSampling --> AppendSampling
AddSampling --> AddCycle
AppendSampling --> AddCycle
AddCycle --> AppendCycle
AddCycle --> AddTemplate
AppendCycle --> AddTemplate
AddTemplate --> AppendTemplate
AddTemplate --> Execute
AppendTemplate --> Execute
```

**Sources:** [protenix/web_service/colab_request_parser.py L463-L496](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L463-L496)

---

## Visualizing Predictions

The `PredictionLoader` class facilitates loading and processing prediction results for visualization.

### Prediction Loading Workflow

```mermaid
flowchart TD

LoaderInit["init(pred_fpath)"]
LoadCIF["_load_cif()"]
LoadConf["_load_confidence_pred()"]
ConvertNP["_convert_to_numpy()"]
GlobCIF["glob.glob(*.cif)"]
ReadCIF["get_structure()"]
GlobJSON["glob.glob(*.json)"]
FilterSummary["Filter summary_confidence"]
FilterFull["Filter full_data_sample"]

LoadCIF --> GlobCIF
GlobCIF --> ReadCIF
LoadConf --> GlobJSON
GlobJSON --> FilterSummary
GlobJSON --> FilterFull
FilterSummary --> ConvertNP
FilterFull --> ConvertNP

subgraph subGraph0 ["PredictionLoader [protenix/web_service/prediction_visualization.py]"]
    LoaderInit
    LoadCIF
    LoadConf
    ConvertNP
    LoaderInit --> LoadCIF
    LoaderInit --> LoadConf
end
```

| Function | File Pointer | Purpose |
| --- | --- | --- |
| `_load_cif` | [protenix/web_service/prediction_visualization.py L56-L64](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py#L56-L64) | Loads all `.cif` structure files from the prediction directory. |
| `_load_confidence_pred` | [protenix/web_service/prediction_visualization.py L66-L89](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py#L66-L89) | Loads summary and full confidence metrics from JSON files. |
| `plot_contact_maps_from_pred` | [protenix/web_service/prediction_visualization.py L91-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py#L91-L125) | Generates adjacency matrices and plots contact maps for structures. |
| `plot_confidence_measures_from_pred` | [protenix/web_service/prediction_visualization.py L128-L212](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py#L128-L212) | Plots pLDDT, PDE, and PAE metrics for quality assessment. |

**Sources:** [protenix/web_service/prediction_visualization.py L30-L212](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/prediction_visualization.py#L30-L212)

---

## Local ColabFold Integration

The system also provides scripts to run MSA searches using a local ColabFold installation instead of the web service.

### Local Search Flow

```mermaid
flowchart TD

RunSearch["run_colabfold_search()"]
ProcessA3M["A3MProcessor"]
SplitSeq["split_sequences()"]
Config["LocalColabFoldConfig"]
MMseqsLocal["colabfold_search command"]
RawA3M["Merged A3M file"]
OutFiles["pairing.a3m<br>non_pairing.a3m"]

Config --> RunSearch
RunSearch --> MMseqsLocal
MMseqsLocal --> RawA3M
RawA3M --> ProcessA3M
SplitSeq --> OutFiles

subgraph scripts/colabfold_msa.py ["scripts/colabfold_msa.py"]
    RunSearch
    ProcessA3M
    SplitSeq
    ProcessA3M --> SplitSeq
end
```

The `A3MProcessor` class handles splitting the merged A3M output from ColabFold into the pairing and non-pairing formats required by Protenix [scripts/colabfold_msa.py L104-L143](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/colabfold_msa.py#L104-L143)

**Sources:** [scripts/colabfold_msa.py L22-L190](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/colabfold_msa.py#L22-L190)

 [scripts/colabfold_msa.py L191-L209](https://github.com/bytedance/Protenix/blob/c3bfc365/scripts/colabfold_msa.py#L191-L209)

---

## Key Configuration Constants

### Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MMSEQS_SERVICE_HOST_URL` | `https://protenix-server.com/api/msa` | MMseqs2 service endpoint [protenix/web_service/colab_request_parser.py L38-L40](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L38-L40) |
| `PROTENIX_ROOT_DIR` | `$HOME` | Root directory for data and checkpoints [protenix/web_service/colab_request_parser.py L44](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L44-L44) |

### Path Constants

| Constant | Value | Purpose |
| --- | --- | --- |
| `DATA_CACHE_DIR` | `{PROTENIX_ROOT_DIR}/common/` | Location for reference data [protenix/web_service/colab_request_parser.py L46](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L46-L46) |
| `CHECKPOINT_DIR` | `{PROTENIX_ROOT_DIR}/checkpoint/` | Location for model checkpoints [protenix/web_service/colab_request_parser.py L47](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L47-L47) |

**Sources:** [protenix/web_service/colab_request_parser.py L38-L48](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/web_service/colab_request_parser.py#L38-L48)