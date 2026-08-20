# Performance Evaluation

> **Relevant source files**
> * [assets/12l_md_templates.md](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1)
> * [assets/6uof_A_animation.gif](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/6uof_A_animation.gif)

This document covers the systematic evaluation and comparison of different AlphaFlow model variants, focusing on performance metrics, benchmarking methodologies, and comparative analysis across model architectures. For information about analyzing individual protein ensembles, see [Ensemble Analysis](/bjing2016/alphaflow/7.1-ensemble-analysis).

## Overview

Performance evaluation in the AlphaFlow system involves comparing model variants across multiple dimensions including structural accuracy, ensemble diversity, dynamic properties, and computational efficiency. The evaluation framework supports comparison between different layer counts (48-layer vs 12-layer), training approaches (base vs distilled), and model types (PDB vs MD vs MD+Templates).

## Performance Metrics Framework

The AlphaFlow evaluation system employs a comprehensive set of metrics to assess both structural and dynamic properties of generated ensembles:

### Performance Metrics Overview

```mermaid
flowchart TD

RUNTIME["Runtime Performance<br>Inference Speed"]
SCALABILITY["Model Scalability<br>Batch Processing"]
MEMORY["Memory Usage<br>Resource Consumption"]
WEAK_J["Weak contacts J<br>Long-range Interactions"]
MI_RHO["Exposed MI matrix ρ<br>Mutual Information"]
TRANS_J["Transient contacts J<br>Dynamic Contacts"]
EXPOSED_J["Exposed residue J<br>Surface Properties"]
RMSD["Pairwise RMSD<br>Structure Similarity"]
RMSF["All-atom RMSF<br>Flexibility Analysis"]
RMSF_CORR["RMSF Correlation<br>Global & Per-target"]
W2_ROOT["Root mean W2-dist<br>Ensemble Diversity"]
W2_MD["MD PCA W2-dist<br>MD Reference"]
W2_JOINT["Joint PCA W2-dist<br>Combined Analysis"]
PC_SIM["PC-similarity<br>Principal Components"]

RMSD --> W2_ROOT
RMSF --> W2_ROOT

subgraph subGraph1 ["Dynamic Metrics"]
    W2_ROOT
    W2_MD
    W2_JOINT
    PC_SIM
    W2_MD --> PC_SIM
end

subgraph subGraph0 ["Structural Metrics"]
    RMSD
    RMSF
    RMSF_CORR
end

subgraph subGraph3 ["Computational Metrics"]
    RUNTIME
    SCALABILITY
    MEMORY
    RUNTIME --> SCALABILITY
end

subgraph subGraph2 ["Contact Analysis"]
    WEAK_J
    MI_RHO
    TRANS_J
    EXPOSED_J
    WEAK_J --> MI_RHO
end
```

Sources: [assets/12l_md_templates.md L1-L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L1-L17)

## Model Architecture Comparison

The performance evaluation framework supports systematic comparison between different model architectures, particularly focusing on the trade-offs between accuracy and computational efficiency.

### Architecture Performance Matrix

```mermaid
flowchart TD

BASE_48["48l Base<br>High Accuracy<br>38s runtime"]
DIST_48["48l Distilled<br>Balanced Performance<br>3.8s runtime"]
BASE_12["12l Base<br>Good Accuracy<br>15.2s runtime"]
DIST_12["12l Distilled<br>Fast Inference<br>1.56s runtime"]
ACCURACY["Structural Accuracy<br>RMSD: 2.18 → 1.40<br>RMSF: 1.31 → 0.76"]
SPEED["Inference Speed<br>38s → 1.56s<br>24x speedup"]
DIVERSITY["Ensemble Diversity<br>W2-dist: 1.95 → 2.43<br>Reduced diversity"]

BASE_48 --> ACCURACY
DIST_12 --> SPEED
BASE_48 --> DIVERSITY
DIST_12 --> DIVERSITY

subgraph subGraph2 ["Performance Trade-offs"]
    ACCURACY
    SPEED
    DIVERSITY
end

subgraph subGraph1 ["12-Layer Models"]
    BASE_12
    DIST_12
end

subgraph subGraph0 ["48-Layer Models"]
    BASE_48
    DIST_48
end
```

Sources: [assets/12l_md_templates.md L1-L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L1-L17)

## Benchmarking Results

The following table presents comprehensive benchmarking results comparing 48-layer and 12-layer AlphaFlow-MD+Templates models in both base and distilled configurations:

| Metric | 48l (base) | 48l (distilled) | 12l (base) | 12l (distilled) |
| --- | --- | --- | --- | --- |
| **Structural Accuracy** |  |  |  |  |
| Pairwise RMSD | 2.18 | 1.73 | 1.94 | 1.40 |
| Pairwise RMSD r | 0.94 | 0.92 | 0.81 | 0.76 |
| All-atom RMSF | 1.31 | 1.00 | 1.01 | 0.76 |
| Global RMSF r | 0.91 | 0.89 | 0.78 | 0.74 |
| Per-target RMSF r | 0.90 | 0.88 | 0.89 | 0.86 |
| **Dynamic Properties** |  |  |  |  |
| Root mean W₂-dist | 1.95 | 2.18 | 2.26 | 2.43 |
| MD PCA W₂-dist | 1.25 | 1.41 | 1.40 | 1.56 |
| Joint PCA W₂-dist | 1.58 | 1.68 | 1.78 | 1.90 |
| % PC-sim > 0.5 | 44% | 43% | 46% | 39% |
| **Contact Analysis** |  |  |  |  |
| Weak contacts J | 0.62 | 0.51 | 0.60 | 0.56 |
| Transient contacts J | 0.47 | 0.42 | 0.36 | 0.24 |
| Exposed residue J | 0.50 | 0.47 | 0.47 | 0.44 |
| Exposed MI matrix ρ | 0.25 | 0.18 | 0.21 | 0.13 |
| **Performance** |  |  |  |  |
| Runtime (s) | 38 | 3.8 | 15.2 | 1.56 |

Sources: [assets/12l_md_templates.md L2-L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L2-L17)

## Runtime Performance Analysis

The evaluation framework demonstrates significant performance improvements through model distillation and layer reduction:

### Runtime Scaling Analysis

```mermaid
flowchart TD

SLOWEST["48l Base<br>38.0s<br>Highest Accuracy"]
FAST["12l Base<br>15.2s<br>2.5x faster"]
FASTER["48l Distilled<br>3.8s<br>10x faster"]
FASTEST["12l Distilled<br>1.56s<br>24x faster"]
HIGH_ACC["High Accuracy<br>RMSD: 1.40-2.18<br>RMSF: 0.76-1.31"]
MED_ACC["Medium Accuracy<br>Correlation: 0.76-0.94"]
DIVERSITY["Ensemble Diversity<br>W2-dist: 1.95-2.43"]

SLOWEST --> HIGH_ACC
FAST --> MED_ACC
FASTER --> MED_ACC
FASTEST --> DIVERSITY

subgraph subGraph1 ["Accuracy Impact"]
    HIGH_ACC
    MED_ACC
    DIVERSITY
end

subgraph subGraph0 ["Performance Hierarchy"]
    SLOWEST
    FAST
    FASTER
    FASTEST
    SLOWEST --> FASTEST
    FAST --> FASTEST
end
```

Sources: [assets/12l_md_templates.md L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L17-L17)

## Evaluation Methodology

The performance evaluation process involves systematic comparison across multiple model configurations using standardized test sets and metrics:

### Evaluation Pipeline

```mermaid
flowchart TD

MODELS["Model Variants<br>48l/12l base/distilled"]
TEST_SET["Test Dataset<br>Protein sequences"]
PARAMS["Evaluation Parameters<br>Diffusion steps, samples"]
PREDICT["predict.py<br>Ensemble Generation"]
ENSEMBLES["Generated Ensembles<br>Multiple conformations"]
REFERENCE["Reference Data<br>MD trajectories, PDB"]
ANALYZE["analyze_ensembles.py<br>Metric Computation"]
STRUCTURAL["Structural Metrics<br>RMSD, RMSF analysis"]
DYNAMIC["Dynamic Metrics<br>PCA, Wasserstein"]
CONTACT["Contact Analysis<br>Distance matrices"]
PRINT["print_analysis.py<br>Report Generation"]
COMPARISON["Comparative Analysis<br>Performance tables"]
VISUALIZE["Visualization<br>Performance plots"]

MODELS --> PREDICT
TEST_SET --> PREDICT
PARAMS --> PREDICT
ENSEMBLES --> ANALYZE
REFERENCE --> ANALYZE
STRUCTURAL --> PRINT
DYNAMIC --> PRINT
CONTACT --> PRINT

subgraph subGraph3 ["Reporting Phase"]
    PRINT
    COMPARISON
    VISUALIZE
    PRINT --> COMPARISON
    PRINT --> VISUALIZE
end

subgraph subGraph2 ["Analysis Phase"]
    ANALYZE
    STRUCTURAL
    DYNAMIC
    CONTACT
    ANALYZE --> STRUCTURAL
    ANALYZE --> DYNAMIC
    ANALYZE --> CONTACT
end

subgraph subGraph1 ["Generation Phase"]
    PREDICT
    ENSEMBLES
    REFERENCE
    PREDICT --> ENSEMBLES
end

subgraph subGraph0 ["Input Configuration"]
    MODELS
    TEST_SET
    PARAMS
end
```

The evaluation methodology incorporates both absolute performance metrics and relative comparisons against molecular dynamics reference data to assess the quality of generated conformational ensembles.

Sources: [assets/12l_md_templates.md L1-L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L1-L17)

## Performance Trade-off Analysis

The benchmarking results reveal clear trade-offs between computational efficiency and ensemble quality:

* **Accuracy vs Speed**: The 12-layer distilled model achieves 24x speedup (1.56s vs 38s) while maintaining reasonable structural accuracy
* **Ensemble Diversity**: Faster models show reduced ensemble diversity (higher Wasserstein distances) indicating less conformational sampling
* **Contact Prediction**: Transient contact prediction accuracy decreases significantly in distilled models (0.47 → 0.24 for 12-layer)
* **Correlation Preservation**: RMSF correlations remain relatively stable across model variants, suggesting preserved flexibility patterns

This analysis enables users to select appropriate model configurations based on their specific requirements for accuracy, speed, and ensemble diversity.

Sources: [assets/12l_md_templates.md L2-L17](https://github.com/bjing2016/alphaflow/blob/02dc0376/assets/12l_md_templates.md?plain=1#L2-L17)