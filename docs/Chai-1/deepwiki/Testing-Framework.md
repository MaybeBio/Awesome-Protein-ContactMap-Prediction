# Testing Framework

> **Relevant source files**
> * [.github/workflows/pytest.yml](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml)
> * [chai_lab/data/dataset/inference_dataset.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/inference_dataset.py)
> * [chai_lab/data/parsing/input_validation.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/input_validation.py)
> * [tests/__init__.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/__init__.py)
> * [tests/example_inputs.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/example_inputs.py)
> * [tests/test_inference_dataset.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py)
> * [tests/test_kalign.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_kalign.py)
> * [tests/test_msa_a3m_tokenization.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_msa_a3m_tokenization.py)
> * [tests/test_msa_preprocess.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_msa_preprocess.py)
> * [tests/test_parsing.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py)
> * [tests/test_restraints.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py)

This document outlines the testing infrastructure and practices used in the Chai-1 molecular structure prediction system. For information about the type system used in conjunction with testing, see [Type System](/chaidiscovery/chai-lab/9.1-type-system).

## Overview

The Chai-1 codebase uses pytest as its primary testing framework to verify component functionality and ensure system reliability. Tests cover input parsing, molecular conformer generation, tool integration, dataset collation, and restraint handling. All tests are integrated into the development workflow through automated continuous integration processes.

```mermaid
flowchart TD

TestFramework["Chai-1 Testing Framework"]
pytest["pytest"]
TestFiles["Test Files<br>(tests/*.py)"]
CI["CI Integration<br>(GitHub Actions)"]
AutoExec["Automated Test Execution"]
ParsingTests["test_parsing.py<br>FASTA & entity validation"]
InferenceDatasetTests["test_inference_dataset.py<br>Chain loading & tokenization"]
RestraintTests["test_restraints.py<br>Contact & pocket restraints"]
MSATests["test_msa_*.py<br>A3M tokenization & pairing"]
ExampleInputs["example_inputs.py<br>Test data fixtures"]

TestFramework --> pytest
pytest --> TestFiles
TestFramework --> CI
CI --> AutoExec
TestFiles --> ParsingTests
TestFiles --> InferenceDatasetTests
TestFiles --> RestraintTests
TestFiles --> MSATests
TestFiles --> ExampleInputs
```

Sources: [.github/workflows/pytest.yml L1-L31](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L1-L31)

 [tests/test_parsing.py L1-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L1-L80)

 [tests/test_inference_dataset.py L1-L124](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L1-L124)

 [tests/test_restraints.py L1-L126](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L1-L126)

## Test Directory Structure

Tests are organized in the `tests/` directory at the repository root. The testing framework follows standard pytest conventions:

1. Test files follow the naming convention `test_*.py` for automatic discovery.
2. Each test file focuses on testing a specific component or module.
3. Test functions use the `test_` prefix naming convention.
4. Shared test data is organized in helper modules like `example_inputs.py` [tests/example_inputs.py L6-L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/example_inputs.py#L6-L38)


Sources: [tests/test_parsing.py L1-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L1-L80)

 [tests/test_inference_dataset.py L1-L124](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L1-L124)

 [tests/test_restraints.py L1-L126](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L1-L126)

 [tests/example_inputs.py L1-L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/example_inputs.py#L1-L38)

## Continuous Integration

Tests run automatically via GitHub Actions as defined in the `pytest.yml` workflow configuration [.github/workflows/pytest.yml L1-L31](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L1-L31)

 The CI pipeline is triggered on pushes to `main` and all pull requests.

```mermaid
flowchart TD

CodeChange["Code Change<br>(Push to main or PR)"]
GitHubActions["GitHub Actions<br>pytest.yml workflow"]
UbuntuSetup["Setup Ubuntu Environment"]
SysDeps["Install System Dependencies<br>sudo apt-get install kalign"]
PythonSetup["Setup Python 3.10<br>with pip cache"]
TorchInstall["Install CPU-only PyTorch<br>--extra-index-url pytorch.org"]
UVInstall["Install dependencies with uv<br>filter out nvidia packages"]
RunTests["Run pytest ./tests/"]
TestResult["Tests Pass?"]
WorkflowSuccess["Workflow Succeeds"]
WorkflowFail["Workflow Fails"]

CodeChange --> GitHubActions
GitHubActions --> UbuntuSetup
UbuntuSetup --> SysDeps
SysDeps --> PythonSetup
PythonSetup --> TorchInstall
TorchInstall --> UVInstall
UVInstall --> RunTests
RunTests --> TestResult
TestResult --> WorkflowSuccess
TestResult --> WorkflowFail
```

### CI Configuration Details

| Configuration | Value | Purpose |
| --- | --- | --- |
| Python Version | 3.10 | Ensures compatibility with minimum supported version [.github/workflows/pytest.yml L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/ .github/workflows/pytest.yml#L19-L19) |
| PyTorch Version | CPU-only | Allows testing in GPU-free environments [.github/workflows/pytest.yml L24](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L24-L24) |
| Package Manager | uv | Fast dependency resolution and installation [.github/workflows/pytest.yml L24-L26](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L24-L26) |
| System Dependencies | kalign | Required for sequence alignment tests [.github/workflows/pytest.yml L15](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L15-L15) |

Sources: [.github/workflows/pytest.yml L1-L31](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L1-L31)

## Test Implementation Patterns

### Input Parsing and Entity Identification

The `test_parsing.py` module verifies that the system correctly identifies molecular types and parses FASTA headers.

* **Entity Types**: Validates `EntityType` identification for proteins, DNA, RNA, and ligands [tests/test_parsing.py L45-L57](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L45-L57)
* **Modified Sequences**: Ensures `constituents_of_modified_fasta` correctly handles bracketed residues like `(KCJ)` or `(SEP)` [tests/test_parsing.py L24-L28](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L24-L28)
* **SMILES**: Confirms FASTA-formatted SMILES strings are correctly read [tests/test_parsing.py L73-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L73-L80)

### Inference Dataset and Tokenization

The `test_inference_dataset.py` module tests the conversion of raw inputs into `Chain` objects and `AllAtomStructureContext`.

* **Malformed Inputs**: Verifies that malformed SMILES (e.g., "Zn" instead of "[Zn+2]") are dropped during chain loading [tests/test_inference_dataset.py L29-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L29-L43)
* **Ions**: Ensures ions like `[Mg+2]` carry the correct charge and atomic number (element 12) in the structure context [tests/test_inference_dataset.py L52-L61](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L52-L61)
* **Merging**: Tests the `AllAtomStructureContext.merge` function to ensure multiple chains (proteins and ligands) are correctly combined into a single context [tests/test_inference_dataset.py L80-L82](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L80-L82)

### Restraints and Feature Collation

The `test_restraints.py` module tests the end-to-end flow from restraint parsing to feature generation.

* **Chain Mapping**: `test_restraints_with_manual_chain_names` verifies that restraints are correctly mapped to tokens regardless of whether automatic (A, B, C) or manual subchain IDs are used [tests/test_restraints.py L49-L126](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L49-L126)
* **Feature Generation**: It uses the `Collate` class and `chai1.feature_factory` to ensure that `TokenDistanceRestraint` and `TokenPairPocketRestraint` features are populated in the batch [tests/test_restraints.py L105-L116](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L105-L116)

Sources: [tests/test_parsing.py L24-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L24-L80)

 [tests/test_inference_dataset.py L29-L107](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L29-L107)

 [tests/test_restraints.py L49-L126](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L49-L126)

## Running Tests Locally

To run tests locally, follow the setup pattern used in CI:

1. **Install system dependencies**: ```markdown sudo apt-get install kalign  # On Ubuntu/Debian ```
2. **Install dependencies** (using `uv` for speed, as seen in CI): ``` uv pip install -r requirements.inuv pip install -r requirements.devuv pip install -e . ```
3. **Run pytest**: ``` pytest ./tests/ ```

### Targeted Testing

```markdown
# Run specific test filepytest tests/test_inference_dataset.py # Run specific test functionpytest tests/test_inference_dataset.py::test_ions_parsing # Run with verbose outputpytest ./tests/ -v
```

Sources: [.github/workflows/pytest.yml L22-L30](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/.github/workflows/pytest.yml#L22-L30)

 [tests/test_inference_dataset.py L52-L61](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L52-L61)

## Key Test Components

| Component | Test File | Description |
| --- | --- | --- |
| FASTA Parsing | `test_parsing.py` | Validates `read_fasta` and `constituents_of_modified_fasta` [tests/test_parsing.py L59-L71](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L59-L71) |
| Dataset Loading | `test_inference_dataset.py` | Tests `load_chains_from_raw` and `AllAtomResidueTokenizer` [tests/test_inference_dataset.py L12-L26](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L12-L26) |
| Restraints | `test_restraints.py` | Tests `parse_pairwise_table` and restraint feature mapping [tests/test_restraints.py L31-L38](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L31-L38) |
| MSA Processing | `test_msa_a3m_tokenization.py` | Tests `tokenize_sequences_to_arrays` and insertion handling [tests/test_msa_a3m_tokenization.py L10-L37](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_msa_a3m_tokenization.py#L10-L37) |
| Alignment | `test_kalign.py` | Tests `kalign_query_to_reference` tool integration [tests/test_kalign.py L5-L17](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_kalign.py#L5-L17) |

Sources: [tests/test_parsing.py L1-L80](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_parsing.py#L1-L80)

 [tests/test_inference_dataset.py L1-L124](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_inference_dataset.py#L1-L124)

 [tests/test_restraints.py L1-L126](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_restraints.py#L1-L126)

 [tests/test_msa_a3m_tokenization.py L1-L37](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_msa_a3m_tokenization.py#L1-L37)

 [tests/test_kalign.py L1-L17](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_kalign.py#L1-L17)