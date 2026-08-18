---
title: "Protenix Ecosystem"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/1.1-protenix-ecosystem
---
# Protenix Ecosystem

# Protenix Ecosystem

> **Relevant source files**
> - [README\.md](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1)
> - [assets/protenix\_base\_default\_v1\.0\.0\_metrics2\.png](https://github.com/bytedance/Protenix/blob/c3bfc365/assets/protenix_base_default_v1.0.0_metrics2.png)
> - [docs/PTX\_V1\_Technical\_Report\_202602042356\.pdf](https://github.com/bytedance/Protenix/blob/c3bfc365/docs/PTX_V1_Technical_Report_202602042356.pdf)

## Purpose and Scope

 This document describes the broader Protenix ecosystem, including downstream applications and complementary tools built on or alongside the core Protenix structure prediction system\. The ecosystem consists of:

 - **PXDesign**: De novo protein\-binder design suite built on the foundation model\.
- **PXMeter**: Evaluation toolkit and benchmark datasets for reproducible results\.
- **Protenix\-Dock**: Classical protein\-ligand docking framework using empirical scoring\.
- **Protenix Web Server**: Unified web interface for structure prediction and design\.

 For information about the core Protenix model architecture and inference system, see [Model Architecture](https://github.com/bytedance/Protenix/blob/c3bfc365/Model Architecture) and [Inference System](https://github.com/bytedance/Protenix/blob/c3bfc365/Inference System) For training the foundation model, see [Training System](https://github.com/bytedance/Protenix/blob/c3bfc365/Training System)

 **Sources**: [README\.md?plain=1 L23-L28](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L23-L28) [README\.md?plain=1 L31-L42](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L31-L42)

---

## Ecosystem Architecture

 The Protenix ecosystem follows a layered architecture where the core foundation model enables multiple downstream applications while being evaluated by independent benchmarking infrastructure\.

  **Ecosystem Component Relationships**
 The diagram shows how Protenix serves as the foundation model that enables PXDesign for protein design applications, is evaluated by PXMeter benchmarks, and is complemented by Protenix\-Dock's classical approach\. Both CLI and web interfaces provide access to the core capabilities\.

 **Sources**: [README\.md?plain=1 L23-L28](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L23-L28) [README\.md?plain=1 L63-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L63-L73)

---

## PXDesign: Protein\-Binder Design

### Overview

 PXDesign is a model suite for de novo protein\-binder design built on the Protenix foundation model\. It achieves experimentally validated success rates of 20\-73% across multiple protein targets, representing 2\-6× improvement over prior state\-of\-the\-art methods including AlphaProteo and RFdiffusion\.

 **Sources**: [README\.md?plain=1 L24](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L24-L24)

### Performance Characteristics

| Metric | PXDesign | Prior SOTA \(AlphaProteo/RFdiffusion\) | Improvement |
| --- | --- | --- | --- |
| Experimental Success Rate | 20\-73% | ~10\-30% | 2\-6× |
| Foundation Model | Protenix \(v1/v2\) | Varied architectures | Built on AF3 reproduction |
| Access Method | Protenix Server | Various platforms | Unified interface |

 The experimental success rates indicate the percentage of designed binders that successfully bind to target proteins in wet\-lab validation experiments\.

 **Sources**: [README\.md?plain=1 L24](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L24-L24)

### Foundation Model Dependency

  **PXDesign Foundation Model Usage**
 PXDesign leverages trained Protenix checkpoints as the foundation for its design framework, particularly utilizing the constraint features and structure prediction capabilities\.

 **Sources**: [README\.md?plain=1 L24](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L24-L24) [README\.md?plain=1 L37-L39](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L37-L39)

### Access and Availability

 PXDesign is freely accessible through the Protenix Web Server at `protenix-server.com` and documented at `protenix.github.io/pxdesign`\.

 **Sources**: [README\.md?plain=1 L4](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L4-L4) [README\.md?plain=1 L24](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L24-L24)

---

## PXMeter: Evaluation Toolkit

### Overview

 PXMeter is an open\-source toolkit designed for reproducible evaluation of biomolecular structure prediction models\. It provides manually curated benchmark datasets and standardized evaluation metrics to enable fair comparison across different prediction systems\.

 **Repository**: `github.com/bytedance/PXMeter`

 **Sources**: [README\.md?plain=1 L26](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L26-L26)

### Benchmark Datasets

 PXMeter includes four primary benchmark suites, each targeting different aspects of structure prediction:

| Benchmark | Source | Focus Area | Notes |
| --- | --- | --- | --- |
| PoseBusters v2 | PoseBusters project | Protein\-ligand complexes | Manually reviewed for artifacts |
| AF3 Nucleic Acid Complexes | AlphaFold 3 paper | DNA/RNA interactions | Based on Nature 2024 publication |
| AF3 Antibody Set | AlphaFold 3 metadata | Antibody\-antigen binding | Metadata from AF3 repository |
| Recent PDB | Curated by PXMeter team | Recently deposited structures | Removes experimental artifacts |

 All datasets undergo manual review to remove experimental artifacts and non\-biological interactions\.

 **Sources**: [README\.md?plain=1 L26](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L26-L26) [README\.md?plain=1 L91-L93](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L91-L93)

### Protenix Evaluation Methodology

 The benchmark evaluation of Protenix follows rigorous inference protocols to ensure statistical robustness:

| Parameter | Value | Purpose |
| --- | --- | --- |
| Model Seeds | 5 to 1000 | Statistical robustness vs\. efficiency |
| Total Predictions | Varies by seed count | Comprehensive sampling |
| Ranking Method | Ranking score | Best structure selection |
| Training Cut\-off Date | 2021\-09\-30 | Alignment with AlphaFold 3 |

 **Sources**: [README\.md?plain=1 L71](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L71-L71) [README\.md?plain=1 L84-L86](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L84-L86)

---

## Protenix\-Dock: Classical Docking Framework

### Overview

 Protenix\-Dock is a classical protein\-ligand docking framework that uses empirical scoring functions without neural networks\. It provides a complementary approach to neural structure prediction for rigid docking tasks\.

 **Repository**: `github.com/bytedance/Protenix-Dock`

 **Sources**: [README\.md?plain=1 L28](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L28-L28)

### Architecture Characteristics

| Aspect | Protenix\-Dock | Protenix \(Neural\) |
| --- | --- | --- |
| Approach | Classical/Empirical | Deep Learning |
| Scoring Functions | Physics\-based potentials | Learned representations |
| Neural Networks | None | 368M \- 464M parameters |
| Docking Type | Rigid docking | Flexible prediction |
| Computational Cost | Low | High \(GPU required\) |

 **Sources**: [README\.md?plain=1 L28](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L28-L28) [README\.md?plain=1 L65-L70](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L65-L70)

---

## Protenix Web Server

### Overview

 The Protenix Web Server \(`protenix-server.com`\) provides a unified web interface for both structure prediction and protein\-binder design, integrating the core Protenix model with PXDesign capabilities\.

 **Sources**: [README\.md?plain=1 L4](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L4-L4) [README\.md?plain=1 L24](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L24-L24)

### Service Architecture

  **Protenix Web Server Architecture**
 The web server integrates request processing, MSA generation, prediction services, and result formatting into a unified interface accessible via browser\.

 **Sources**: [README\.md?plain=1 L4](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L4-L4) [README\.md?plain=1 L41-L42](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L41-L42)

---

## Ecosystem Integration Points

### Model Checkpoint Sharing

 All ecosystem components rely on the same trained Protenix model checkpoints:

| Checkpoint | Parameters | Ecosystem Usage |
| --- | --- | --- |
| protenix\-v2 | 464M | SOTA antibody\-antigen, ligand plausibility |
| protenix\_base\_default\_v1\.0\.0 | 368M | Core predictions, AF3 baseline |
| protenix\_base\_20250630\_v1\.0\.0 | 368M | Practical application scenarios |
| protenix\_base\_default\_v0\.5\.0 | 368M | Backward compatibility |

 **Sources**: [README\.md?plain=1 L63-L73](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L63-L73)

### Community Resources

| Resource | Purpose | Link |
| --- | --- | --- |
| GitHub Repository | Source code, issues, PRs | github\.com/bytedance/Protenix |
| Technical Report \(v1\) | Protenix\-v1 architecture/benchmarks | PTX\_V1\_Technical\_Report\_202602042356\.pdf |
| Technical Report \(v2\) | Protenix\-v2 updates | PX2\.pdf |
| Twitter | Announcements, updates | @ai4s\_protenix |
| Slack | Community discussion | Protenix Workspace |

 **Sources**: [README\.md?plain=1 L4-L14](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L4-L14) [README\.md?plain=1 L31-L33](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L31-L33)

---

## Summary

 The Protenix ecosystem demonstrates a comprehensive approach to biomolecular structure prediction and design:

 - **Protenix** serves as the trainable foundation model \(up to 464M parameters\)\.
- **PXDesign** enables de novo protein\-binder design with high success rates\.
- **PXMeter** provides standardized evaluation with manually curated benchmarks\.
- **Protenix\-Dock** offers classical docking as a complementary approach\.
- **Protenix Web Server** unifies prediction and design in a browser\-accessible interface\.

 This ecosystem architecture separates concerns while enabling tight integration: the foundation model remains focused on structure prediction, evaluation remains independent for credibility, and design applications build on proven foundations\.

 **Sources**: [README\.md?plain=1 L1-L94](https://github.com/bytedance/Protenix/blob/c3bfc365/README.md?plain=1#L1-L94)

---
*Source: [https://deepwiki.com/bytedance/Protenix/1.1-protenix-ecosystem](https://deepwiki.com/bytedance/Protenix/1.1-protenix-ecosystem) on DeepWiki*