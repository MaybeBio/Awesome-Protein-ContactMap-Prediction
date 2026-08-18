# Testing Framework

> **Relevant source files**
> * [.github/workflows/ci.yaml](https://github.com/google-deepmind/alphafold3/blob/97639fff/.github/workflows/ci.yaml)
> * [run_alphafold_data_test.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py)
> * [run_alphafold_test.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py)
> * [src/alphafold3/test_data/featurised_example.json](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/featurised_example.json)

This document provides an overview of the testing infrastructure used in AlphaFold 3. The testing framework ensures correctness and reliability by verifying both end-to-end functionality and individual components through automated tests with golden reference data. For detailed information on test organization and CI/CD, see [Test Infrastructure](/google-deepmind/alphafold3/7.1-test-infrastructure). For validation approaches, see [Test Data and Validation](/google-deepmind/alphafold3/7.2-test-data-and-validation).

## Overview

The AlphaFold 3 testing framework consists of two primary test suites implemented in Python using the `absltest` framework:

1. **End-to-End Inference Tests** (`run_alphafold_test.py`) - Validates complete pipeline execution from input to structure prediction [run_alphafold_test.py L11](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L11-L11)
2. **Data Pipeline Tests** (`run_alphafold_data_test.py`) - Verifies MSA generation, template search, and featurization [run_alphafold_data_test.py L11](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L11-L11)

Tests compare outputs against golden reference data using RMSD calculations, hash verification, and text diffs.

**Test Suite Architecture**

```mermaid
flowchart TD

RAT["run_alphafold_test.py<br>InferenceTest"]
RADT["run_alphafold_data_test.py<br>DataPipelineTest"]
MiniDB["miniature_databases/<br>BFD, UniRef90, PDB, etc."]
FeatExample["featurised_example.pkl/json"]
GoldenOutputs["alphafold_run_outputs/<br>*.pkl golden files"]
RunAlphaFold["run_alphafold.py<br>process_fold_input()"]
ModelRunner["ModelRunner<br>run_inference()"]
DataPipeline["pipeline.DataPipeline"]
Featurisation["featurisation.py"]

RAT --> FeatExample
RAT --> GoldenOutputs
RAT --> RunAlphaFold
RAT --> ModelRunner
RADT --> MiniDB
RADT --> DataPipeline
RADT --> Featurisation

subgraph SystemUnderTest ["System Under Test"]
    RunAlphaFold
    ModelRunner
    DataPipeline
    Featurisation
    RunAlphaFold --> ModelRunner
    RunAlphaFold --> DataPipeline
    DataPipeline --> Featurisation
end

subgraph TestData ["test_data/"]
    MiniDB
    FeatExample
    GoldenOutputs
end

subgraph TestFiles ["Test Files"]
    RAT
    RADT
end
```

Sources: [run_alphafold_test.py L66-L154](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L66-L154)

 [run_alphafold_data_test.py L100-L207](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L100-L207)

## Test Data Organization

Test data is located in `test_data/` and includes miniature databases (subsampled to ~1000 sequences each), pre-featurized examples, and golden outputs for regression testing. The miniature databases enable fast test execution while maintaining coverage of key functionality [run_alphafold_test.py L71-L105](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L71-L105)

**Test Data Structure**

```mermaid
flowchart TD

DefaultPKL["run_alphafold_test_output_bucket_default.pkl"]
Bucket1024PKL["run_alphafold_test_output_bucket_1024.pkl"]
FeatJSON["featurised_example.json"]
FeatPKL["featurised_example.pkl"]
BFD["bfd-first_non_consensus_sequences__subsampled_1000.fasta"]
MGnify["mgy_clusters__subsampled_1000.fa"]
UniProt["uniprot_all__subsampled_1000.fasta"]
UniRef90["uniref90__subsampled_1000.fasta"]
NTRNA["nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta"]
Rfam["rfam_14_4_clustered_rep_seq__subsampled_1000.fasta"]
RNACentral["rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta"]
PDBSeqres["pdb_seqres_2022_09_28__subsampled_1000.fasta"]
PDBMMCIF["pdb_mmcif/"]

subgraph TestDataDir ["test_data/"]

subgraph GoldenOutputs ["alphafold_run_outputs/"]
    DefaultPKL
    Bucket1024PKL
end

subgraph PreFeaturized ["Pre-Featurized Data"]
    FeatJSON
    FeatPKL
end

subgraph MiniDB ["miniature_databases/"]
    BFD
    MGnify
    UniProt
    UniRef90
    NTRNA
    Rfam
    RNACentral
    PDBSeqres
    PDBMMCIF
end
end
```

Sources: [run_alphafold_test.py L71-L105](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L71-L105)

 [run_alphafold_data_test.py L106-L140](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L106-L140)

 [src/alphafold3/test_data/featurised_example.json L1-L67](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/featurised_example.json#L1-L67)

## End-to-End Inference Testing

The `InferenceTest` class in `run_alphafold_test.py` validates the complete pipeline from input to predicted structures. Tests use a sample input (PDB ID 5tgy with ligand 7BU) and verify outputs against golden reference data stored as pickled `InferenceResult` objects [run_alphafold_test.py L124-L144](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L124-L144)

**InferenceTest Class Structure**

```mermaid
classDiagram
    class InferenceTest {
        -DataPipelineConfig _data_pipeline_config
        -ModelConfig _model_config
        -ModelRunner _runner
        -str _test_input_json
        +setUp() : void
        +test_model_inference() : void
        +test_process_fold_input_runs_only_inference() : void
        +test_inference(bucket, seed) : void
    }
    class ModelRunner {
        +run_inference(batch, rng_key) : dict
        +extract_inference_results(batch, result, target_name) : list
        +extract_embeddings(result, num_tokens) : tuple
    }
    InferenceTest --> ModelRunner : "uses"
```

**Test Coverage**

The test suite includes:

* **`test_model_inference()`**: Validates `ModelRunner.run_inference()` [run_alphafold.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold.py)  on pre-featurized inputs and checks embeddings extraction [run_alphafold_test.py L156-L175](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L156-L175)
* **`test_process_fold_input_runs_only_inference()`**: Verifies error handling when MSAs are missing if the data pipeline is not configured [run_alphafold_test.py L177-L186](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L177-L186)
* **`test_inference()`**: Full end-to-end test with parameterized bucket sizes (default, 1024), validates results against expected values [run_alphafold_test.py L188-L200](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L188-L200)

Sources: [run_alphafold_test.py L66-L200](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L66-L200)

## Data Pipeline Testing

The `DataPipelineTest` class validates MSA generation, template search, and featurization without requiring GPU resources. Tests use hash-based comparison of complex nested data structures to detect regressions [run_alphafold_data_test.py L55-L86](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L55-L86)

**DataPipelineTest Class Structure**

```mermaid
classDiagram
    class DataPipelineTest {
        -DataPipelineConfig _data_pipeline_config
        -str _test_input_json
        +setUp() : void
        +compare_golden(result_path) : void
        +test_config() : void
        +test_featurisation() : void
    }
    class DataPipeline {
        +process(input) : dict
    }
    class hash_data {
        «singledispatch»
        +_hash_data(x) : str
    }
    DataPipelineTest --> DataPipeline : "invokes"
    DataPipelineTest --> hash_data : "uses for comparison"
```

**Test Coverage**

* **`test_config()`**: Validates `make_model_config()` generates correct configuration [run_alphafold_data_test.py L194-L201](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L194-L201)
* **`test_featurisation()`**: Tests `pipeline.DataPipeline.process()` and compares featurized output hashes against golden references [run_alphafold_data_test.py L203-L207](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L203-L207)

Sources: [run_alphafold_data_test.py L100-L207](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L100-L207)

## Validation Strategies

The testing framework employs multiple validation strategies depending on the data type:

**Validation Methods**

| Method | Used For | Implementation | Tolerance |
| --- | --- | --- | --- |
| RMSD Comparison | Predicted structures | `alignment.rmsd_from_coords()` | < 3.0Å (full), < 1.4Å (high confidence) |
| Hash Verification | Featurized data | `_hash_data()` singledispatch | Exact match |
| Text Diff | JSON/config files | `difflib.unified_diff()` | Exact match |

**Regression Detection Workflow**

```mermaid
flowchart TD

TestRun["Test Execution"]
GenerateOutput["Generate Test Output"]
LoadGolden["Load Golden Reference"]
CheckType["Data Type?"]
RMSDCalc["Calculate RMSD<br>alignment.rmsd_from_coords()"]
HashCalc["Calculate Hash<br>_hash_data()"]
DiffCalc["Generate Diff<br>_generate_diff()"]
RMSDThreshold["RMSD < threshold?"]
HashMatch["Hashes match?"]
DiffEmpty["Diff empty?"]
Pass["Test Pass"]

TestRun --> GenerateOutput
GenerateOutput --> LoadGolden
LoadGolden --> CheckType
CheckType --> RMSDCalc
CheckType --> HashCalc
CheckType --> DiffCalc
RMSDCalc --> RMSDThreshold
HashCalc --> HashMatch
DiffCalc --> DiffEmpty
RMSDThreshold --> Pass
HashMatch --> Pass
DiffEmpty --> Pass
```

Sources: [run_alphafold_test.py L54-L63](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L54-L63)

 [run_alphafold_data_test.py L55-L98](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L55-L98)

 [run_alphafold_data_test.py L180-L192](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L180-L192)

## Test Execution

Tests can be run locally or via CI. Local execution requires HMMER binaries and test data:

```markdown
# Run data pipeline tests (CPU only, CI-compatible)uv run python run_alphafold_data_test.py # Run inference tests (requires GPU and model parameters)uv run python run_alphafold_test.py
```

The CI pipeline (`.github/workflows/ci.yaml`) automatically runs `run_alphafold_data_test.py` on push/PR to verify data pipeline functionality [.github/workflows/ci.yaml L43-L44](https://github.com/google-deepmind/alphafold3/blob/97639fff/ .github/workflows/ci.yaml#L43-L44)

**Test Configuration Parameters**

| Parameter | Test File | Purpose | Example Value |
| --- | --- | --- | --- |
| `jackhmmer_binary_path` | Both | Path to jackhmmer executable | `shutil.which('jackhmmer')` |
| `model_dir` | InferenceTest | Directory containing model parameters | `run_alphafold.MODEL_DIR.value` |
| `flash_attention_implementation` | Both | Flash attention backend | `'triton'` |

Sources: [run_alphafold_test.py L38-L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L38-L42)

 [run_alphafold_test.py L145-L154](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L145-L154)

 [run_alphafold_data_test.py L141-L157](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L141-L157)

 [.github/workflows/ci.yaml L34-L44](https://github.com/google-deepmind/alphafold3/blob/97639fff/.github/workflows/ci.yaml#L34-L44)

## Continuous Integration

The AlphaFold 3 testing framework is integrated with a CI pipeline that automatically runs the data pipeline tests on code changes:

```mermaid
flowchart TD

PR["Pull Request / Push"]
CI["CI Workflow"]
Setup["Setup Python & Dependencies"]
InstallHMMER["Install HMMER Tools"]
BuildData["Build Test Data"]
RunTests["Run Data Pipeline Tests"]
Results["Test Results"]

PR --> CI
CI --> Setup
Setup --> InstallHMMER
InstallHMMER --> BuildData
BuildData --> RunTests
RunTests --> Results
```

The CI workflow currently runs `run_alphafold_data_test.py` to verify data pipeline functionality without requiring GPUs, making it suitable for standard CI environments [.github/workflows/ci.yaml L43-L44](https://github.com/google-deepmind/alphafold3/blob/97639fff/.github/workflows/ci.yaml#L43-L44)

Sources: [.github/workflows/ci.yaml L1-L45](https://github.com/google-deepmind/alphafold3/blob/97639fff/.github/workflows/ci.yaml#L1-L45)

## Implementation Details

### Hash-Based Verification

For complex data structures that are difficult to compare directly, the test framework implements a hash-based verification approach using `functools.singledispatch` [run_alphafold_data_test.py L54-L55](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L54-L55)

 This handles various types including `jax.Array`, `np.ndarray`, and custom objects like `Structure` [run_alphafold_data_test.py L61-L86](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L61-L86)

### Diff Generation

The framework uses `difflib.unified_diff` to highlight differences between expected and actual outputs [run_alphafold_test.py L54-L63](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L54-L63)

 [run_alphafold_data_test.py L88-L97](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_data_test.py#L88-L97)

## Related Pages

* [Test Infrastructure](/google-deepmind/alphafold3/7.1-test-infrastructure) — Detailed documentation of CI/CD and execution.
* [Test Data and Validation](/google-deepmind/alphafold3/7.2-test-data-and-validation) — Deep dive into RMSD validation and golden files.
* [Data Pipeline](/google-deepmind/alphafold3/4.2-data-pipeline) — Details on the pipeline components being tested.
* [Featurization](/google-deepmind/alphafold3/4.3-featurization) — Information about the featurization process verified in tests.
* [Model Inference](/google-deepmind/alphafold3/4.4-model-inference) — Details on the inference process validated by the test framework.