# Test Data and Validation

> **Relevant source files**
> * [run_alphafold_test.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py)
> * [src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_1024.pkl](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_1024.pkl)
> * [src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_default.pkl](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_default.pkl)
> * [src/alphafold3/test_data/featurised_example.pkl](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/featurised_example.pkl)
> * [src/alphafold3/test_data/miniature_databases/pdb_mmcif/5y2e.cif](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/miniature_databases/pdb_mmcif/5y2e.cif)
> * [src/alphafold3/test_data/miniature_databases/pdb_mmcif/6s61.cif](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/miniature_databases/pdb_mmcif/6s61.cif)
> * [src/alphafold3/test_data/miniature_databases/pdb_mmcif/6ydw.cif](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/miniature_databases/pdb_mmcif/6ydw.cif)
> * [src/alphafold3/test_data/miniature_databases/pdb_mmcif/7rye.cif](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/miniature_databases/pdb_mmcif/7rye.cif)

This document describes the test data infrastructure and validation strategies used to ensure correctness of AlphaFold 3 predictions. It covers the organization of test data files, the use of miniature databases for testing, golden output storage, and the RMSD-based validation approach for regression testing.

For information about the test infrastructure and CI/CD pipeline, see [Test Infrastructure](/google-deepmind/alphafold3/7.1-test-infrastructure).

## Purpose and Scope

The testing framework uses a combination of miniature genetic databases and pre-computed golden outputs to validate end-to-end inference results. The primary test case exercises the complete prediction pipeline using a protein-ligand complex (PDB ID: 5tgy), validating both the structural predictions and confidence metrics against expected values.

Sources: [run_alphafold_test.py L11-L66](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L11-L66)

## Test Data Organization

The test data is organized into three main categories: miniature databases for MSA and template searches, featurized examples for model inference testing, and golden outputs for regression validation.

```mermaid
flowchart TD

MiniatureDB["miniature_databases/"]
Featurized["featurised_example.pkl"]
GoldenOutputs["alphafold_run_outputs/"]
BFD["bfd-first_non_consensus_sequences__subsampled_1000.fasta"]
MGnify["mgy_clusters__subsampled_1000.fa"]
UniProt["uniprot_all__subsampled_1000.fasta"]
UniRef90["uniref90__subsampled_1000.fasta"]
NTRNA["nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta"]
Rfam["rfam_14_4_clustered_rep_seq__subsampled_1000.fasta"]
RNACentral["rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta"]
PDB["pdb_mmcif/"]
Seqres["pdb_seqres_2022_09_28__subsampled_1000.fasta"]
DefaultBucket["run_alphafold_test_output_bucket_default.pkl"]
Bucket1024["run_alphafold_test_output_bucket_1024.pkl"]

MiniatureDB --> BFD
MiniatureDB --> MGnify
MiniatureDB --> UniProt
MiniatureDB --> UniRef90
MiniatureDB --> NTRNA
MiniatureDB --> Rfam
MiniatureDB --> RNACentral
MiniatureDB --> PDB
MiniatureDB --> Seqres
GoldenOutputs --> DefaultBucket
GoldenOutputs --> Bucket1024

subgraph Golden ["alphafold_run_outputs/"]
    DefaultBucket
    Bucket1024
end

subgraph Databases ["miniature_databases/"]
    BFD
    MGnify
    UniProt
    UniRef90
    NTRNA
    Rfam
    RNACentral
    PDB
    Seqres
end

subgraph TestData ["test_data/"]
    MiniatureDB
    Featurized
    GoldenOutputs
end
```

**Test Data Directory Structure**

All test data is accessed through the `testing_data.Data` interface, which resolves paths relative to `resources.ROOT`.

Sources: [run_alphafold_test.py L69-L106](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L69-L106)

 [run_alphafold_test.py L28](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L28-L28)

## Miniature Databases

### Database Configuration

The test suite uses miniature versions of the full genetic databases, each subsampled to 1,000 sequences. This enables fast test execution while maintaining realistic data pipeline behavior.

| Database | Purpose | File Pattern | Template Date |
| --- | --- | --- | --- |
| Small BFD | Protein MSA | `bfd-first_non_consensus_sequences__subsampled_1000.fasta` | - |
| MGnify | Protein MSA | `mgy_clusters__subsampled_1000.fa` | - |
| UniProt | Protein MSA | `uniprot_all__subsampled_1000.fasta` | - |
| UniRef90 | Protein MSA | `uniref90__subsampled_1000.fasta` | - |
| NT RNA | RNA MSA | `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta` | - |
| Rfam | RNA MSA | `rfam_14_4_clustered_rep_seq__subsampled_1000.fasta` | - |
| RNACentral | RNA MSA | `rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta` | - |
| PDB mmCIF | Templates | `pdb_mmcif/` | 2021-09-30 |
| PDB Seqres | Templates | `pdb_seqres_2022_09_28__subsampled_1000.fasta` | 2021-09-30 |

### DataPipelineConfig Initialization

The test setup creates a `DataPipelineConfig` with paths to all miniature databases and HMMER tool binaries:

```mermaid
flowchart TD

ResolveDB["Resolve database paths<br>via testing_data.Data"]
BinaryPaths["Locate HMMER binaries<br>shutil.which()"]
CreateConfig["Create DataPipelineConfig"]
JackhmmerBin["jackhmmer_binary_path"]
NhmmerBin["nhmmer_binary_path"]
HmmalignBin["hmmalign_binary_path"]
HmmsearchBin["hmmsearch_binary_path"]
HmmbuildBin["hmmbuild_binary_path"]
DBPaths["Database paths<br>(9 databases)"]
MaxDate["max_template_date<br>2021-09-30"]

subgraph Config ["alphafold3.data.pipeline.DataPipelineConfig"]
    JackhmmerBin
    NhmmerBin
    HmmalignBin
    HmmsearchBin
    HmmbuildBin
    DBPaths
    MaxDate
end

subgraph TestSetup ["InferenceTest.setUp()"]
    ResolveDB
    BinaryPaths
    CreateConfig
    ResolveDB --> CreateConfig
    BinaryPaths --> CreateConfig
end
```

**DataPipelineConfig Initialization for Tests**

Sources: [run_alphafold_test.py L69-L123](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L69-L123)

 [run_alphafold_test.py L107-L123](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L107-L123)

## Test Input Specification

### 5tgy Test Case

The primary test case uses PDB structure 5tgy, a protein-ligand complex consisting of a 110-residue protein chain and a ligand (7BU, 7-tert-butyl-3-methylbenzo[c][1,2,5]thiadiazole).

```mermaid
flowchart TD

LigandID["id: 'LL'"]
CCDCodes["ccdCodes: ['7BU']"]
ProteinID["id: 'P'"]
ProteinSeq["sequence: 110 residues<br>SEFEKLRQTGDELVQA..."]
Modifications["modifications: []"]
UnpairedMSA["unpairedMsa: None"]
PairedMSA["pairedMsa: None"]
Name["name: '5tgy'"]
Seeds["modelSeeds: [1234]"]
Sequences["sequences"]
Dialect["dialect: 'alphafold3'"]
Version["version: 1"]

subgraph Metadata ["Input Metadata"]
    Dialect
    Version
end

subgraph TestInput ["Test Input JSON"]
    Name
    Seeds
    Sequences
end

subgraph Sequence2 ["sequences[1]"]
    LigandID
    CCDCodes
end

subgraph Sequence1 ["sequences[0]"]
    ProteinID
    ProteinSeq
    Modifications
    UnpairedMSA
    PairedMSA
end
```

**Test Input JSON Structure for 5tgy**

The test input is initially provided without MSAs or templates (set to `None`). The data pipeline fills these in during execution, and the enriched input is validated in the output.

Sources: [run_alphafold_test.py L124-L144](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L124-L144)

## Golden Outputs and Pickle Format

### Storage Strategy

Golden outputs are stored as pickled Python dictionaries containing the complete inference results. Each golden output file corresponds to a specific test configuration (bucket size, seed).

```mermaid
flowchart TD

RunInference["run_alphafold.process_fold_input()"]
ExtractResults["Extract ResultsForSeed"]
ConvertDict["Convert to dict:<br>seed, inference_results,<br>full_fold_input"]
Serialize["Pickle and save"]
DefaultPkl["run_alphafold_test_output_bucket_default.pkl"]
Bucket1024Pkl["run_alphafold_test_output_bucket_1024.pkl"]
SeedField["seed: int"]
InferenceResults["inference_results: List[alphafold3.model.model.InferenceResult]"]
FullFoldInput["full_fold_input: alphafold3.common.folding_input.Input"]

subgraph PickleContent ["Pickle Content (List[Dict])"]
    SeedField
    InferenceResults
    FullFoldInput
end

subgraph Storage ["test_data/alphafold_run_outputs/"]
    DefaultPkl
    Bucket1024Pkl
end

subgraph Generation ["Golden Output Generation"]
    RunInference
    ExtractResults
    ConvertDict
    Serialize
    RunInference --> ExtractResults
    ExtractResults --> ConvertDict
    ConvertDict --> Serialize
end
```

**Golden Output Pickle Format**

Each pickle file contains a list of dictionaries with three keys:

* `seed`: The random seed used for inference.
* `inference_results`: List of `InferenceResult` objects (one per sample).
* `full_fold_input`: The enriched `Input` object with MSAs and templates.

Sources: [run_alphafold_test.py L351-L361](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L351-L361)

 [run_alphafold_test.py L366-L377](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L366-L377)

 [src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_default.pkl L1](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/alphafold_run_outputs/run_alphafold_test_output_bucket_default.pkl#L1-L1)

## Validation Pipeline

### End-to-End Test Flow

The validation pipeline runs the complete prediction workflow and compares outputs against golden references using multiple validation criteria.

```mermaid
flowchart TD

InitConfig["Initialize DataPipelineConfig<br>with miniature databases"]
CreateInput["Create test Input<br>(5tgy protein-ligand)"]
LoadGolden["Load golden output<br>from pickle file"]
RunPipeline["run_alphafold.process_fold_input()"]
DataPipe["Run data pipeline<br>(MSA + templates)"]
Featurize["Featurize enriched input"]
ModelInfer["Model inference<br>(5 samples per seed)"]
CheckFiles["Verify output files exist"]
ValidateEmbeddings["Validate embeddings shape<br>(num_tokens, 384)"]
ValidateDistogram["Validate distogram shape<br>(num_tokens, num_tokens, 64)"]
CheckRanking["Verify ranking scores<br>in range [0.66, 0.78]"]
CompareMetadata["Compare metadata<br>(token_chain_ids)"]
ComputeRMSD["Compute full RMSD"]
MaskLowConf["Mask atoms with<br>b_factor < 80.0"]
ComputeMaskedRMSD["Compute masked RMSD"]
CheckThresholds["Check RMSD thresholds:<br>Full < 3.0 Å<br>Masked < 1.4 Å"]

InitConfig --> RunPipeline
CreateInput --> RunPipeline

subgraph StructuralValidation ["Structural Validation"]
    CompareMetadata
    ComputeRMSD
    MaskLowConf
    ComputeMaskedRMSD
    CheckThresholds
end

subgraph OutputValidation ["Output Structure Validation"]
    CheckFiles
    ValidateEmbeddings
    ValidateDistogram
    CheckRanking
end

subgraph Execution ["Execute Inference"]
    RunPipeline
    DataPipe
    Featurize
    ModelInfer
    RunPipeline --> DataPipe
    DataPipe --> Featurize
    Featurize --> ModelInfer
end

subgraph Setup ["Test Setup"]
    InitConfig
    CreateInput
    LoadGolden
end
```

**Complete Validation Pipeline Flow**

Sources: [run_alphafold_test.py L188-L427](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L188-L427)

## RMSD-Based Validation

### Coordinate Comparison Strategy

The test suite employs a two-tier RMSD validation approach: full structure comparison and masked comparison focusing on high-confidence regions.

```mermaid
flowchart TD

ActualCoords["actual_inf.predicted_structure.coords"]
ExpectedCoords["expected_inf.predicted_structure.coords"]
ComputeFull["alphafold3.model.scoring.alignment.rmsd_from_coords(<br>decoy_coords=actual_coords,<br>gt_coords=expected_coords)"]
ThresholdFull["Threshold: < 3.0 Å"]
ExtractBFactor["Extract b_factor values"]
CreateMask["mask = b_factor > 80.0"]
CheckProportion["Verify mask proportion > 0.7"]
ComputeMasked["alphafold3.model.scoring.alignment.rmsd_from_coords(<br>decoy_coords=actual_coords,<br>gt_coords=expected_coords,<br>include_idxs=mask)"]
ThresholdMasked["Threshold: < 1.4 Å"]

ActualCoords --> ComputeFull
ExpectedCoords --> ComputeFull
ActualCoords --> ExtractBFactor
ExpectedCoords --> ComputeMasked

subgraph MaskedRMSD ["Masked RMSD Validation"]
    ExtractBFactor
    CreateMask
    CheckProportion
    ComputeMasked
    ThresholdMasked
    ExtractBFactor --> CreateMask
    CreateMask --> CheckProportion
    CreateMask --> ComputeMasked
    ComputeMasked --> ThresholdMasked
end

subgraph FullRMSD ["Full RMSD Validation"]
    ComputeFull
    ThresholdFull
    ComputeFull --> ThresholdFull
end

subgraph Inputs ["Input Coordinates"]
    ActualCoords
    ExpectedCoords
end
```

**Two-Tier RMSD Validation Strategy**

### Validation Thresholds

| Metric | Threshold | Purpose |
| --- | --- | --- |
| Full RMSD | < 3.0 Å | Ensures overall structural similarity across all atoms |
| Masked RMSD | < 1.4 Å | Validates high-confidence regions (b-factor > 80.0) |
| Mask Proportion | > 70% | Ensures sufficient high-confidence predictions |
| Ranking Score | [0.66, 0.78] | Validates confidence metrics are in expected range |

The 5tgy test case is chosen because it produces stable predictions with low RMSD variance across different seeds, bucket sizes, and device types.

Sources: [run_alphafold_test.py L379-L426](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L379-L426)

 [run_alphafold_test.py L30](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L30-L30)

## Output Structure Validation

### File and Directory Organization

The test validates the complete output directory structure, ensuring all expected files and subdirectories are present.

```mermaid
flowchart TD

DataJSON["{name}_data.json"]
RankingCSV["{name}_ranking_scores.csv"]
TermsOfUse["TERMS_OF_USE.md"]
TopCIF["{name}_model.cif"]
TopConf["{name}_confidences.json"]
TopSummary["{name}_summary_confidences.json"]
DistogramNPZ["{name}_seed-{seed}_distogram.npz"]
DistogramArray["distogram: (num_tokens, num_tokens, 64)"]
EmbeddingsNPZ["{name}_seed-{seed}_embeddings.npz"]
SingleEmbed["single_embeddings: (num_tokens, 384)"]
PairEmbed["pair_embeddings: (num_tokens, num_tokens, 128)"]
SampleCIF["{name}_seed-{seed}_sample-{i}_model.cif"]
SampleConf["{name}_seed-{seed}_sample-{i}_confidences.json"]
SampleSummary["{name}_seed-{seed}_sample-{i}_summary_confidences.json"]
Samples["Sample Directories"]
Embeddings["Embeddings Directory"]
Distogram["Distogram Directory"]
TopFiles["Top-Ranked Files"]
Metadata["Metadata Files"]

subgraph OutputDir ["output_dir/"]
    Samples
    Embeddings
    Distogram
    TopFiles
    Metadata
end

subgraph MetadataFiles ["Metadata"]
    DataJSON
    RankingCSV
    TermsOfUse
end

subgraph TopRanked ["Top-Ranked Output"]
    TopCIF
    TopConf
    TopSummary
end

subgraph DistogramDir ["seed-{seed}_distogram/"]
    DistogramNPZ
    DistogramArray
    DistogramNPZ --> DistogramArray
end

subgraph EmbeddingsDir ["seed-{seed}_embeddings/"]
    EmbeddingsNPZ
    SingleEmbed
    PairEmbed
    EmbeddingsNPZ --> SingleEmbed
    EmbeddingsNPZ --> PairEmbed
end

subgraph SampleDirs ["seed-{seed}_sample-{0-4}/"]
    SampleCIF
    SampleConf
    SampleSummary
end
```

**Expected Output Directory Structure**

### Validation Checks

The test performs the following structural validations:

1. **Directory Structure**: Verifies 5 sample directories, 1 embeddings directory, and 1 distogram directory exist.
2. **File Presence**: Checks each sample directory contains CIF, confidences JSON, and summary JSON files.
3. **Embeddings**: Validates shape `(num_tokens, 384)` for single and `(num_tokens, num_tokens, 128)` for pair embeddings, with `float16` dtype.
4. **Distogram**: Validates shape `(num_tokens, num_tokens, 64)` with `float16` dtype.
5. **Ranking CSV**: Verifies 5 rows with correct seed and sample indices.
6. **Data JSON**: Confirms MSAs, templates, and input sequences are properly saved.

Sources: [run_alphafold_test.py L220-L346](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L220-L346)

## Regression Testing Approach

### Parameterized Test Matrix

The test suite uses parameterized testing to validate consistency across different execution configurations:

```mermaid
flowchart TD

Bucket2["bucket = 1024"]
Seed2["seed = 42"]
Bucket1["bucket = None"]
Seed1["seed = 1"]
Bucket["bucket"]
Seed["seed"]
Stable["Same RMSD thresholds<br>regardless of config"]
Deterministic["Deterministic results<br>for given seed"]
BucketIndependent["Bucketing doesn't affect<br>structural accuracy"]

subgraph Validation ["Validation Expectations"]
    Stable
    Deterministic
    BucketIndependent
end

subgraph Config2 ["Test: bucket_1024"]
    Bucket2
    Seed2
end

subgraph Config1 ["Test: default_bucket"]
    Bucket1
    Seed1
end

subgraph Parameters ["Test Parameters"]
    Bucket
    Seed
end
```

**Parameterized Test Configuration Matrix**

### Golden Output Updates

When code changes affect numerical results, golden outputs must be regenerated:

1. Run the test with output generation enabled (writes to `absltest.TEST_TMPDIR`).
2. The test automatically serializes `ResultsForSeed` objects to pickle format.
3. Copy the generated pickle file to `test_data/alphafold_run_outputs/`.
4. The test will now validate against the new golden outputs.

This approach enables:

* **Regression detection**: Any unintended changes to predictions are catchable.
* **Intentional updates**: Improvements to the model can be validated and accepted by updating golden outputs.
* **Cross-platform consistency**: Ensures predictions are reproducible across different hardware and configurations.

Sources: [run_alphafold_test.py L188-L199](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L188-L199)

 [run_alphafold_test.py L348-L361](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L348-L361)

## Metadata Validation

### Token Chain ID Verification

The test validates that predicted structures maintain correct chain assignments throughout the prediction pipeline:

```mermaid
flowchart TD

ProteinChain["Protein chain 'P'<br>110 residues"]
LigandChain["Ligand chain 'LL'<br>41 tokens"]
TokenChainIDs["metadata['token_chain_ids']"]
ProteinTokens["['P'] * 110"]
LigandTokens["['LL'] * 41"]

ProteinChain --> TokenChainIDs
LigandChain --> TokenChainIDs
TokenChainIDs --> ProteinTokens
TokenChainIDs --> LigandTokens

subgraph Expected ["Expected Assignment"]
    ProteinTokens
    LigandTokens
end

subgraph Prediction ["Predicted Structure"]
    TokenChainIDs
end

subgraph Input ["Input Chains"]
    ProteinChain
    LigandChain
end
```

**Token Chain ID Validation**

The test also verifies:

* **Atom Occupancy**: All atoms have occupancy value of 1.0.
* **Data JSON Enrichment**: The output `_data.json` file contains populated MSAs, templates, and sequences.
* **Terms of Use**: The `TERMS_OF_USE.md` file is correctly generated.

Sources: [run_alphafold_test.py L388-L396](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L388-L396)

## Featurized Example Testing

### Model-Only Inference

A separate test validates model inference without running the full data pipeline, using pre-computed featurized inputs:

```mermaid
flowchart TD

FeaturizedPkl["test_data/featurised_example.pkl"]
LoadPickle["Load featurised_examples"]
ExtractBatch["Extract featurised_example[0]"]
RunInference["run_alphafold.ModelRunner.run_inference(<br>featurised_example,<br>jax.random.PRNGKey(0))"]
ExtractResults["run_alphafold.ModelRunner.extract_inference_results()"]
ExtractEmbeddings["run_alphafold.ModelRunner.extract_embeddings()"]
CheckResult["Assert result is not None"]
CheckEmbedLen["Assert len(embeddings) == 2"]

LoadPickle --> ExtractBatch
ExtractResults --> CheckResult
ExtractEmbeddings --> CheckEmbedLen

subgraph Validation ["Validation"]
    CheckResult
    CheckEmbedLen
end

subgraph ModelTest ["Model Inference Test"]
    ExtractBatch
    RunInference
    ExtractResults
    ExtractEmbeddings
    ExtractBatch --> RunInference
    RunInference --> ExtractResults
    RunInference --> ExtractEmbeddings
end

subgraph PreComputed ["Pre-Computed Data"]
    FeaturizedPkl
    LoadPickle
    FeaturizedPkl --> LoadPickle
end
```

**Featurized Example Testing Flow**

This test validates:

* Model can load and process pre-featurized inputs.
* Inference produces valid results.
* Embeddings are correctly extracted (returns 2 arrays: single and pair).

This enables faster iteration when debugging model-specific issues without re-running the expensive data pipeline.

Sources: [run_alphafold_test.py L156-L176](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L156-L176)

 [src/alphafold3/test_data/featurised_example.pkl L1](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/test_data/featurised_example.pkl#L1-L1)

## Staged Execution Testing

### Data Pipeline vs. Inference Separation

The test framework validates that the system correctly handles staged execution, where data pipeline and inference can be run separately:

```mermaid
flowchart TD

InputNoMSA2["Input without MSAs/templates"]
WithPipelineConfig["data_pipeline_config = DataPipelineConfig"]
Success["Data pipeline enriches input<br>Inference succeeds"]
InputNoMSA["Input without MSAs/templates"]
NoPipelineConfig["data_pipeline_config = None"]
ExpectFailure["Expect ValueError:<br>'missing unpaired MSA'"]

subgraph WithDataPipeline ["Test: With Data Pipeline"]
    InputNoMSA2
    WithPipelineConfig
    Success
    InputNoMSA2 --> WithPipelineConfig
    WithPipelineConfig --> Success
end

subgraph NoDataPipeline ["Test: No Data Pipeline"]
    InputNoMSA
    NoPipelineConfig
    ExpectFailure
    InputNoMSA --> NoPipelineConfig
    NoPipelineConfig --> ExpectFailure
end
```

**Staged Execution Validation**

This test ensures the system properly validates inputs and fails fast when MSAs are missing but no data pipeline is configured to generate them.

Sources: [run_alphafold_test.py L177-L186](https://github.com/google-deepmind/alphafold3/blob/97639fff/run_alphafold_test.py#L177-L186)