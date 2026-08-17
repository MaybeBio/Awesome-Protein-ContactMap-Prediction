---
title: "Testing Framework"
source: deepwiki.com
owner: sokrypton
repo: ColabFold
url: https://deepwiki.com/sokrypton/ColabFold/6.2-testing-framework
---
# Testing Framework

# Testing Framework

> **Relevant source files**
> - [colabfold/alphafold/ipsae\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/colabfold/alphafold/ipsae.py)
> - [test\-data/a3m/5AWL1\.a3m](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/a3m/5AWL1.a3m)
> - [test\-data/a3m/6A5J\.a3m](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/a3m/6A5J.a3m)
> - [test\-data/a3m/empty\.a3m](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/a3m/empty.a3m)
> - [test\-data/batch/input/5AWL\_1\.fasta](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/batch/input/5AWL_1.fasta)
> - [test\-data/batch/input/empty\.fasta](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/batch/input/empty.fasta)
> - [test\-data/mmseqs\-api\-reponses/batch\.json](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/mmseqs-api-reponses/batch.json)
> - [test\-data/mmseqs\-api\-reponses/complex\.json](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/mmseqs-api-reponses/complex.json)
> - [tests/mock\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py)
> - [tests/test\_colabfold\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py)
> - [tests/test\_msa\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py)
> - [tests/test\_utils\.py](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py)

## Purpose and Scope

 The ColabFold testing framework is designed to ensure the reliability and correctness of the protein structure prediction pipeline\. This page documents the testing infrastructure, mock objects, test cases, and how to extend the testing framework\. The framework focuses on validating core functionality while avoiding dependencies on external services and expensive computations during routine testing\.

 The testing strategy employs snapshot testing and mocking to provide deterministic validation of the batch processing pipeline, MSA generation, and model prediction components without requiring live API calls or GPU\-intensive computations\.

 Sources: [mock\.py L1-L214](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L1-L214) [test\_msa\.py L1-L40](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L1-L40) [test\_colabfold\.py L1-L190](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L1-L190)

## Testing Infrastructure Overview

  Sources: [mock\.py L1-L214](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L1-L214) [test\_msa\.py L1-L40](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L1-L40) [test\_utils\.py L1-L141](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py#L1-L141)

## Key Components

### Test Fixtures

 The testing framework uses pytest fixtures to set up the test environment:

  The `prediction_test` fixture:

 1. Sets up logging to the `INFO` level and suppresses JAX device search logs [test\_colabfold\.py L27-L29](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L27-L29)
2. Downloads AlphaFold parameters for `alphafold2_multimer_v1` and `alphafold2_ptm` [test\_colabfold\.py L31-L32](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L31-L32)
3. Resets the AlphaFold `SeedMaker` to 0 to ensure deterministic input features [test\_colabfold\.py L36](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L36-L36)
4. Patches `alphafold.model.data.get_model_haiku_params` with a cached version to speed up parameter loading [test\_colabfold\.py L39-L41](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L39-L41)

 Sources: [test\_colabfold\.py L25-L42](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L25-L42)

### Mock Objects

 The framework includes two primary mock objects that replace expensive external dependencies:

#### MockRunModel

 The `MockRunModel` class replaces the AlphaFold model inference to provide deterministic testing without GPU computation:

  Key features:

 - Stores pre\-computed predictions in compressed pickle files \(`model_feat.pkl.xz` and `model_pred.pkl.xz`\) [mock\.py L70-L71](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L70-L71)
- Uses `jnp_to_np()` to recursively convert JAX arrays to NumPy arrays for serialization [mock\.py L32-L39](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L32-L39)
- Implements `cmp_dict()` for deep comparison of nested dictionaries using `np.allclose()` [mock\.py L93-L114](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L93-L114)
- Excludes `msa_feat` and `msa` from comparisons due to non\-deterministic variance between machines [mock\.py L99-L100](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L99-L100)
- Supports snapshot updates via `UPDATE_SNAPSHOTS` or `PRED_TEST` environment variables [mock\.py L73-L85](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L73-L85)

 Sources: [mock\.py L44-L125](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L44-L125)

#### MMseqs2Mock

 The `MMseqs2Mock` class replaces calls to the MMseqs2 remote API with pre\-recorded responses:

  Key features:

 - Loads API responses from JSON files in `test-data/mmseqs-api-reponses/` [mock\.py L138-L142](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L138-L142)
- Matches API requests by configuration parameters including `query`, `use_env`, `use_filter`, and `pairing_strategy` [mock\.py L169-L177](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L169-L177)
- Uses `split_lines()` and `join_lines()` to store multi\-line responses as arrays for JSON readability [mock\.py L213-L228](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L213-L228)
- Updates snapshots via the `UPDATE_SNAPSHOTS` environment variable [mock\.py L191-L208](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L191-L208)

 Sources: [mock\.py L126-L228](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L126-L228)

## Test Case Categories

### Batch Processing Tests

 The primary test function `test_batch` validates the full end\-to\-end pipeline:

 1. **Input**: Uses queries for `5AWL_1` and `6A5J` [test\_colabfold\.py L46](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L46-L46)
2. **Mocking**: Patches `RunModel.predict` and `run_mmseqs2` [test\_colabfold\.py L52-L56](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L52-L56)
3. **Execution**: Calls `run()` with specific recycle counts and model orders [test\_colabfold\.py L57-L64](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L57-L64)
4. **Validation**: - Checks log messages for sequence length, padding, and reranking [test\_colabfold\.py L67-L82](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L67-L82) - Verifies PDB file generation and line counts \(coordinate validation\) [test\_colabfold\.py L85-L100](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L85-L100) - Confirms `config.json` creation [test\_colabfold\.py L101](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L101-L101)

 Sources: [test\_colabfold\.py L45-L102](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_colabfold.py#L45-L102)

### MSA Generation Tests

 `test_get_msa_and_templates` validates MSA generation across different modes:

 - **mmseqs2\_uniref\_env**: Expected 12 alignment lines [test\_msa\.py L11](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L11-L11)
- **mmseqs2\_uniref**: Expected 8 alignment lines [test\_msa\.py L12](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L12-L12)
- **single\_sequence**: Expected 2 alignment lines [test\_msa\.py L13](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L13-L13)

 Sources: [test\_msa\.py L7-L40](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_msa.py#L7-L40)

### Utility and Input Tests

 The `tests/test_utils.py` file covers data parsing and format conversion:

 - **Query Parsing**: Validates `get_queries` for FASTA directories, empty A3M files, and CSV inputs [test\_utils\.py L6-L39](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py#L6-L39)
- **Format Conversion**: Tests `convert_pdb_to_mmcif` by comparing generated CIFs against expected fixtures [test\_utils\.py L64-L74](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py#L64-L74)
- **mmCIF Validation**: Tests `validate_and_fix_mmcif` for missing revision dates or required fields like `_entity_poly_seq.mon_id` [test\_utils\.py L77-L123](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py#L77-L123)

 Sources: [test\_utils\.py L1-L141](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/test_utils.py#L1-L141)

## Environment Variables

| Variable | Purpose |
| --- | --- |
| UPDATE\_SNAPSHOTS | Updates MMseqs2 API responses and model prediction fixtures tests/mock\.py143 tests/mock\.py191 |
| PRED\_TEST | Runs actual AlphaFold model predictions and compares them against stored snapshots tests/mock\.py73 tests/mock\.py120 |

 Sources: [mock\.py L73-L191](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L73-L191)

## Snapshot Testing Mechanism

### Model Prediction Snapshots

 The framework implements a `cmp_dict` function that performs an "allclose" check on nested dictionaries [mock\.py L93-L114](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L93-L114) It uses `jax.tree_util.tree_flatten` to ensure every leaf in the feature/prediction tree matches the snapshot [mock\.py L114](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L114-L114)

### API Response Snapshots

 MMseqs2 responses are stored in `test-data/mmseqs-api-reponses/`\. The `MMseqs2Mock` class ensures that if a query configuration \(e\.g\., `use_pairing`, `filter`\) changes, the test will fail unless snapshots are updated, preventing silent changes in MSA retrieval logic [mock\.py L188-L210](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L188-L210)

 Sources: [mock\.py L93-L210](https://github.com/sokrypton/ColabFold/blob/0c788a0e/tests/mock.py#L93-L210) [batch\.json L1-L40](https://github.com/sokrypton/ColabFold/blob/0c788a0e/test-data/mmseqs-api-reponses/batch.json#L1-L40)

---
*Source: [https://deepwiki.com/sokrypton/ColabFold/6.2-testing-framework](https://deepwiki.com/sokrypton/ColabFold/6.2-testing-framework) on DeepWiki*