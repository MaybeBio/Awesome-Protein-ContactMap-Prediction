---
title: "Inference Pipeline Overview"
source: deepwiki.com
owner: bytedance
repo: Protenix
url: https://deepwiki.com/bytedance/Protenix/3.1-inference-pipeline-overview
---
# Inference Pipeline Overview

# Inference Pipeline Overview

> **Relevant source files**
> - [runner/batch\_inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py)
> - [runner/inference\.py](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py)

 This document describes the end\-to\-end inference workflow in Protenix, from input preparation through prediction generation\. The pipeline transforms raw molecular structure data \(PDB/CIF files or sequences\) into structure predictions with confidence metrics\.

 **Scope**: This page covers the complete inference workflow orchestrated by the CLI and `InferenceRunner` class\. For detailed information on specific stages, see: input conversion \([3\.2](https://github.com/bytedance/Protenix/blob/c3bfc365/3.2)\), MSA generation \([3\.3](https://github.com/bytedance/Protenix/blob/c3bfc365/3.3)\), running predictions \([3\.4](https://github.com/bytedance/Protenix/blob/c3bfc365/3.4)\), output formats \([3\.5](https://github.com/bytedance/Protenix/blob/c3bfc365/3.5)\), and constraint\-guided predictions \([3\.6](https://github.com/bytedance/Protenix/blob/c3bfc365/3.6)\)\.

## Architecture Overview

 The inference system consists of five major stages, orchestrated through the `protenix` CLI and executed by the `InferenceRunner` class:

  **Diagram: End\-to\-End Inference Pipeline with Code Entities**

 This pipeline can be executed via specific CLI commands or as a unified workflow through `protenix pred`\.

 **Sources**: [batch\_inference\.py L70-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L70-L165) [batch\_inference\.py L568-L714](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L568-L714) [inference\.py L64-L237](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L237) [infer\_dataloader\.py L23-L52](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py#L23-L52)

## Pipeline Stages

### Stage 1: Input Conversion

 Raw structural data \(PDB/CIF files\) must be converted to Protenix's JSON format\. The `tojson` command handles this conversion:

| Input Format | Handler | Output |
| --- | --- | --- |
| PDB file | pdb\_to\_cif\(\) → cif\_to\_input\_json\(\) | JSON with sequences |
| CIF file | cif\_to\_input\_json\(\) | JSON with sequences |
| Ligand \(SDF/SMI\) | lig\_file\_to\_atom\_info\(\) | Embedded in JSON |

 The conversion process extracts sequences, ligand structures, covalent bonds, and optional constraints\.

 **Key Parameters**:

 - `--altloc`: Select altloc conformations \("first", "A", "B", etc\.\) [batch\_inference\.py L596-L597](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L596-L597)
- `--assembly_id`: Expand bioassembly structure [batch\_inference\.py L598-L599](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L598-L599)

 **Sources**: [batch\_inference\.py L568-L636](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L568-L636) [json\_maker\.py L28-L112](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/json_maker.py#L28-L112)

### Stage 2: Preprocessing \(MSA & Templates\)

 The `preprocess_input` function [batch\_inference\.py L70-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L70-L165) coordinates search operations for evolutionary and structural information\.

  **Diagram: Preprocessing Workflow**

 1. **Protein MSA**: Orchestrated by `update_infer_json` [msa\_search\.py L78-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/msa_search.py#L78-L125) Supports 'protenix' or 'colabfold' modes\.
2. **Template Search**: Uses `hmmsearch` and `hmmbuild` to find structural templates [batch\_inference\.py L124-L131](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L124-L131)
3. **RNA MSA**: Performs `nhmmer` searches against RNA databases \(nt\-rna, rfam, rna\_central\) [batch\_inference\.py L134-L146](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L134-L146)

 **Sources**: [batch\_inference\.py L70-L165](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/batch_inference.py#L70-L165) [msa\_search\.py L78-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/msa_search.py#L78-L125) [rna\_msa\_search\.py L26-L105](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/rna_msa_search.py#L26-L105)

### Stage 3: Feature Generation

 The `get_inference_dataloader` function [infer\_dataloader\.py L23-L52](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py#L23-L52) builds the data pipeline using `InferenceDataset`\.

| Category | Components |
| --- | --- |
| Token Features | token\_index, asym\_id, entity\_id, residue\_index |
| Reference Features | ref\_pos, ref\_mask, ref\_element |
| MSA/Template Features | msa\_feat, templates |
| Bond Features | bond\_feats, covalent\_bond |

 **Sources**: [infer\_dataloader\.py L23-L52](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/inference/infer_dataloader.py#L23-L52) [featurizer\_inference\.py L38-L124](https://github.com/bytedance/Protenix/blob/c3bfc365/protenix/data/featurizer_inference.py#L38-L124)

### Stage 4: Model Execution

 The `InferenceRunner` class manages the execution lifecycle:

  **Diagram: Model Execution Sequence**

 **Dynamic Configuration**: `update_inference_configs` [inference\.py L440-L454](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L440-L454) adjusts Automatic Mixed Precision \(AMP\) based on the number of tokens to optimize memory usage:

 - `N_token > 3840`: Disable AMP for confidence head and diffusion\.
- `N_token > 2560`: Disable AMP for confidence head\.

 **Sources**: [inference\.py L64-L237](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L237) [inference\.py L440-L454](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L440-L454) [inference\.py L457-L526](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L457-L526)

### Stage 5: Output Generation

 The `DataDumper` class [dumper\.py L43-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L43-L125) processes the model outputs into final files:

 - **CIF Files**: Atomic coordinates for each sample/seed\.
- **Confidence JSON**: Contains `ranking_score`, `plddt`, `pae`, and `pde` metrics\.

 **Sources**: [dumper\.py L43-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L43-L125) [inference\.py L496-L508](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L496-L508)

## Core Classes

 **`InferenceRunner`** [inference\.py L64-L237](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L237)

 - **`init_env`**: Sets up CUDA, NCCL for distributed inference, and kernel compilation paths \(CUTLASS/FastLayerNorm\) [inference\.py L84-L128](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L84-L128)
- **`load_checkpoint`**: Loads weights into the `Protenix` model, handling DDP prefix mapping [inference\.py L144-L185](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L144-L185)
- **`predict`**: Performs the forward pass within `torch.no_grad()` and AMP contexts [inference\.py L214-L237](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L214-L237)

 **`DataDumper`** [dumper\.py L43-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L43-L125)

 - Handles the conversion of raw tensor outputs into biologically relevant formats\.
- Implements ranking logic based on `ranking_score` [dumper\.py L228-L251](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L228-L251)

 **Sources**: [inference\.py L64-L237](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L64-L237) [dumper\.py L43-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/dumper.py#L43-L125)

## Optimization and Hardware Compatibility

 Protenix includes specialized handling for different GPU architectures in `update_gpu_compatible_configs` [inference\.py L534-L555](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L534-L555):

 - **V100/Older GPUs**: Forces `fp32` and `torch` kernels as they lack support for advanced attention kernels [inference\.py L538-L545](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L538-L545)
- **A100\+**: Enables `bf16` and optimized kernels like `triattention` or `deepspeed` [inference\.py L547-L553](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L547-L553)

 **Sources**: [inference\.py L534-L555](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L534-L555) [inference\.py L108-L125](https://github.com/bytedance/Protenix/blob/c3bfc365/runner/inference.py#L108-L125)

## Related Pages

 - For input JSON specification: [4\.1](https://github.com/bytedance/Protenix/blob/c3bfc365/4.1)
- For MSA generation details: [3\.3](https://github.com/bytedance/Protenix/blob/c3bfc365/3.3)
- For output format interpretation: [3\.5](https://github.com/bytedance/Protenix/blob/c3bfc365/3.5)
- For Training\-Free Guidance \(TFG\): [3\.7](https://github.com/bytedance/Protenix/blob/c3bfc365/3.7)

---
*Source: [https://deepwiki.com/bytedance/Protenix/3.1-inference-pipeline-overview](https://deepwiki.com/bytedance/Protenix/3.1-inference-pipeline-overview) on DeepWiki*