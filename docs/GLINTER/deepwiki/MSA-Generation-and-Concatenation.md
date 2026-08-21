# MSA Generation and Concatenation

> **Relevant source files**
> * [external/A3M_NoGap](https://github.com/zw2x/glinter/blob/8871ca11/external/A3M_NoGap)
> * [external/A3M_SpecBloc](https://github.com/zw2x/glinter/blob/8871ca11/external/A3M_SpecBloc)
> * [external/A3M_To_PSI](https://github.com/zw2x/glinter/blob/8871ca11/external/A3M_To_PSI)
> * [external/MSA_ConCat](https://github.com/zw2x/glinter/blob/8871ca11/external/MSA_ConCat)
> * [external/meff_cdhit](https://github.com/zw2x/glinter/blob/8871ca11/external/meff_cdhit)
> * [external/verify_fasta](https://github.com/zw2x/glinter/blob/8871ca11/external/verify_fasta)
> * [preprocess/MSA/concat_msa.sh](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh)
> * [preprocess/MSA/filter_msa.sh](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/filter_msa.sh)
> * [preprocess/MSA/msa_to_hhm.sh](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/msa_to_hhm.sh)
> * [preprocess/MSA/run_hhblits.sh](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh)
> * [scripts/concat_msa.sh](https://github.com/zw2x/glinter/blob/8871ca11/scripts/concat_msa.sh)
> * [scripts/run_msa.sh](https://github.com/zw2x/glinter/blob/8871ca11/scripts/run_msa.sh)

The GLINTER preprocessing pipeline generates evolutionary information through Multiple Sequence Alignments (MSAs). This process involves building per-monomer MSAs using `hhblits`, calculating alignment effectiveness (Meff), and concatenating monomer alignments into paired dimer MSAs (A3M_CC format) to capture co-evolutionary signals across the protein-protein interface.

## Implementation Overview

The MSA generation workflow is implemented as a series of shell scripts that orchestrate external bioinformatics tools. The process is divided into two main stages:

1. **Monomer MSA Generation**: Building individual alignments for the receptor and ligand.
2. **Dimer MSA Concatenation**: Pairing sequences from monomer MSAs based on taxonomic information and filtering the result.

### System Data Flow

The following diagram illustrates the transition from raw sequence data to the concatenated dimer MSA used for feature tensorization.

**Diagram: MSA Generation and Concatenation Data Flow**


## Monomer MSA Generation

Individual MSAs are generated for each monomer using the `run_hhblits.sh` script. This script performs sequence validation, coverage calculation, and iterative searching against the `HHDB` database.

### Key Logic and Parameters

* **Sequence Validation**: The script uses the `verify_fasta` utility to ensure the input sequence is valid for HH-suite [preprocess/MSA/run_hhblits.sh L31-L37](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L31-L37)
* **Dynamic Coverage**: Minimum alignment coverage (`-cov`) is calculated dynamically. It defaults to 60% but is adjusted if the query sequence is short to ensure at least 80 residues are covered [preprocess/MSA/run_hhblits.sh L41-L49](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L41-L49)
* **HHblits Execution**: Runs for 3 iterations with an E-value threshold of 0.001 and specific filtering flags (`-maxfilt 500000 -diff inf -id 99`) [preprocess/MSA/run_hhblits.sh L51-L57](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L51-L57)
* **Meff Calculation**: The `meff_cdhit` utility calculates the number of effective sequences in the MSA. If the MSA exceeds 200,000 lines, Meff is capped at 11 [preprocess/MSA/run_hhblits.sh L68-L75](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L68-L75)

**Sources:** [preprocess/MSA/run_hhblits.sh L1-L81](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L1-L81)

 [scripts/run_msa.sh L1-L5](https://github.com/zw2x/glinter/blob/8871ca11/scripts/run_msa.sh#L1-L5)

## Dimer MSA Concatenation

The `concat_msa.sh` script (found in both `scripts/` and `preprocess/MSA/`) manages the pairing of sequences between the receptor and ligand monomers.

### Implementation Details

The concatenation process uses specialized C++ utilities located in the `external/` directory:

1. **A3M_NoGap**: Removes gaps from the monomer A3M files to prepare for species blocking [preprocess/MSA/concat_msa.sh L10-L16](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L10-L16)
2. **A3M_SpecBloc**: Groups sequences by species using the `TaxTree` database to facilitate taxonomically-aware pairing [preprocess/MSA/concat_msa.sh L11-L17](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L11-L17)
3. **MSA_ConCat**: Performs the actual concatenation of the species-blocked alignments into a single `.a3m_cc` file [preprocess/MSA/concat_msa.sh L21](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L21-L21)
4. **meff_cdhit**: Re-calculates the effectiveness of the concatenated dimer MSA using a 65% sequence identity threshold [preprocess/MSA/concat_msa.sh L22](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L22-L22)

**Sources:** [preprocess/MSA/concat_msa.sh L1-L23](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L1-L23)

 [scripts/concat_msa.sh L1-L5](https://github.com/zw2x/glinter/blob/8871ca11/scripts/concat_msa.sh#L1-L5)

## Filtering and Profile Generation

Once the dimer MSA is concatenated, it undergoes final filtering and profile generation to be used by the ESM-MSA Transformer and structural encoders.

### MSA Filtering

The `filter_msa.sh` script applies `hhfilter` to the concatenated alignment. It uses a diversity threshold (`-diff 200`) and a coverage threshold (`-cov 20`) to produce the final `.hh.a3m` file [preprocess/MSA/filter_msa.sh L7-L9](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/filter_msa.sh#L7-L9)

### HHM Profile Generation

The `msa_to_hhm.sh` script converts monomer A3M files into HHM profiles using `hhmake` [preprocess/MSA/msa_to_hhm.sh L9](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/msa_to_hhm.sh#L9-L9)

 These profiles are then parsed into Python-readable formats by `LoadHHM.py` for downstream feature extraction [preprocess/MSA/msa_to_hhm.sh L10](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/msa_to_hhm.sh#L10-L10)

**Sources:** [preprocess/MSA/filter_msa.sh L1-L12](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/filter_msa.sh#L1-L12)

 [preprocess/MSA/msa_to_hhm.sh L1-L12](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/msa_to_hhm.sh#L1-L12)

## Code Entity Map

The following table maps the natural language steps to the specific code entities and external binaries responsible for their execution.

| Process Step | Code Entity / Script | External Binary / Tool | Output Format |
| --- | --- | --- | --- |
| **Search** | `run_hhblits.sh` | `hhblits` | `.a3m` |
| **Diversity Check** | `run_hhblits.sh` | `meff_cdhit` | `.meff` |
| **Taxonomic Pairing** | `concat_msa.sh` | `A3M_SpecBloc` | `.a3m_sb` |
| **Dimer Join** | `concat_msa.sh` | `MSA_ConCat` | `.a3m_cc` |
| **Final Filter** | `filter_msa.sh` | `hhfilter` | `.hh.a3m` |
| **Profile Building** | `msa_to_hhm.sh` | `hhmake` | `.hhm` |

**Sources:** [preprocess/MSA/run_hhblits.sh L1-L81](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/run_hhblits.sh#L1-L81)

 [preprocess/MSA/concat_msa.sh L1-L23](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/concat_msa.sh#L1-L23)

 [preprocess/MSA/filter_msa.sh L1-L12](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/filter_msa.sh#L1-L12)

 [preprocess/MSA/msa_to_hhm.sh L1-L12](https://github.com/zw2x/glinter/blob/8871ca11/preprocess/MSA/msa_to_hhm.sh#L1-L12)