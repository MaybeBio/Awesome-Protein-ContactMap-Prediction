# Benchmark Dataset

> **Relevant source files**
> * [benchmark/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/README.md?plain=1)
> * [benchmark/bAIes/Ab40/Ab40.pdb](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/Ab40.pdb)
> * [benchmark/bAIes/Ab40/Ab40_nvt.data](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/Ab40_nvt.data)
> * [benchmark/bAIes/Ab40/Ab40_nvt.in](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/Ab40_nvt.in)
> * [benchmark/bAIes/Ab40/atom_list_matrix.ndx](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/atom_list_matrix.ndx)
> * [benchmark/bAIes/Ab40/baies_gauss_matrix.dat](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/baies_gauss_matrix.dat)
> * [benchmark/bAIes/Ab40/dry_ff_20240524_correct.cmap](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/dry_ff_20240524_correct.cmap)
> * [benchmark/bAIes/Ab40/plumed_Ab40.dat](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/plumed_Ab40.dat)
> * [scripts/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1)
> * [tutorial/README.md](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/README.md?plain=1)

This page provides a high-level overview of the benchmark dataset used to validate bAIes-IDP. The benchmark consists of 21 proteins, categorized by their structural characteristics and AlphaFold-2 prediction behavior. We explore three distinct simulation variants: bAIes, bAIes-N, and coil, each serving a specific purpose in understanding IDP ensembles. Detailed information for each simulation variant, including file structures and reproduction steps, is provided in their respective child pages.

## Benchmark Proteins and Categories

The benchmark dataset comprises 21 proteins, carefully selected to represent a range of intrinsically disordered protein (IDP) characteristics and AlphaFold-2 prediction scenarios. These proteins are grouped into four categories: Disordered, Structure motifs, Alphafold-2 prediction errors, and Multidomain. This categorization helps in evaluating the performance of bAIes-IDP across different types of IDPs and prediction challenges. The full list of proteins and their categories can be found in the `benchmark/README.md` file [benchmark/README.md L9-L33](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/README.md?plain=1#L9-L33)

### Benchmark Protein Categories

```mermaid
flowchart TD

A["Benchmark Proteins (21 Total)"]
B["Disordered (6)"]
C["Structure motifs (6)"]
D["Alphafold-2 prediction errors (6)"]
E["Multidomain (3)"]
Ab40["Ab40"]
p61_Hck["p61_Hck"]
emerin67_170["emerin67-170"]
His_PknG1_75["His-PknG1-75"]
Colicin_NT_domain["Colicin_NT_domain"]
Hug1["Hug1"]
PaaA2["PaaA2"]
Nt_SOCS5["Nt-SOCS5"]
Alb3_A3CT["Alb3-A3CT"]
idr_SSRP1["idr_SSRP1"]
UBact["UBact"]
FCP1["FCP1"]
asyn["asyn"]
ACTR["ACTR"]
NHE1["NHE1"]
Nsp2_ctlIDR["Nsp2_ctlIDR"]
spm_FrpC["spm_FrpC"]
drkN["drkN"]
GS8["GS8"]
Ubq2["Ubq2"]
Ubq3["Ubq3"]

A --> B
A --> C
A --> D
A --> E
B --> Ab40
B --> p61_Hck
B --> emerin67_170
B --> His_PknG1_75
B --> Colicin_NT_domain
B --> Hug1
C --> PaaA2
C --> Nt_SOCS5
C --> Alb3_A3CT
C --> idr_SSRP1
C --> UBact
C --> FCP1
D --> asyn
D --> ACTR
D --> NHE1
D --> Nsp2_ctlIDR
D --> spm_FrpC
D --> drkN
E --> GS8
E --> Ubq2
E --> Ubq3
```

Sources: [benchmark/README.md L9-L33](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/README.md?plain=1#L9-L33)

## Simulation Variants

To thoroughly validate bAIes-IDP, three distinct simulation variants are employed for the benchmark proteins. Each variant provides unique insights into the conformational ensembles of IDPs.

### bAIes Benchmark Simulations

The `bAIes` simulations represent the core bAIes-IDP methodology, incorporating AlphaFold-2 derived distance restraints with a Jeffreys prior. This variant aims to generate ensembles that reflect the structural preferences predicted by AlphaFold-2. For details on the file structure, the 21 proteins covered, and how to reproduce these results, see [bAIes Benchmark Simulations](/COSBlab/bAIes-IDP/5.1-baies-benchmark-simulations).

### bAIes-N Benchmark Simulations

The `bAIes-N` simulations are a variant of the `bAIes` approach where the Jeffreys prior is omitted (`PRIOR=NONE`) [benchmark/bAIes/Ab40/plumed_Ab40.dat L3](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/plumed_Ab40.dat#L3-L3)

 This variant is crucial for understanding the impact of the Jeffreys prior on the resulting conformational ensembles and for exploring scenarios where a less biased approach might be preferred. This variant covers 18 of the benchmark proteins. For more information, refer to [bAIes-N Benchmark Simulations](/COSBlab/bAIes-IDP/5.2-baies-n-benchmark-simulations).

### Coil Benchmark Simulations

The `coil` simulations serve as an unbiased reference ensemble, representing the behavior of proteins in a random coil state without any AlphaFold-2 derived restraints. These simulations are essential for comparison with the biased `bAIes` and `bAIes-N` ensembles, highlighting the effect of the AlphaFold-2 bias. This variant covers 18 of the benchmark proteins. For a detailed description, see [Coil Benchmark Simulations](/COSBlab/bAIes-IDP/5.3-coil-benchmark-simulations).

### Benchmark Simulation Workflow Overview


Sources:
[benchmark/bAIes/Ab40/baies_gauss_matrix.dat](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/baies_gauss_matrix.dat)

[benchmark/bAIes/Ab40/plumed_Ab40.dat L3](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/plumed_Ab40.dat#L3-L3)

[benchmark/bAIes/Ab40/Ab40_nvt.data](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/Ab40_nvt.data)

[benchmark/bAIes/Ab40/Ab40_nvt.in](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/Ab40_nvt.in)

[benchmark/bAIes/Ab40/atom_list_matrix.ndx](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/benchmark/bAIes/Ab40/atom_list_matrix.ndx)

[scripts/README.md L12-L15](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/scripts/README.md?plain=1#L12-L15)

[tutorial/README.md L9-L10](https://github.com/COSBlab/bAIes-IDP/blob/16faa7f7/tutorial/README.md?plain=1#L9-L10)