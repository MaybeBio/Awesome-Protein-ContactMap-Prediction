# Usage Examples

> **Relevant source files**
> * [LICENSE](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/LICENSE)
> * [README.md](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1)
> * [example/1GL1_A.fasta](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_A.fasta)
> * [example/1GL1_A_uniref100.a3m](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_A_uniref100.a3m)
> * [example/1GL1_I.fasta](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_I.fasta)
> * [example/1GL1_I_uniref100.a3m](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_I_uniref100.a3m)
> * [predict.py](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py)

This page provides practical examples of how to use PLMGraph-Inter for predicting inter-protein contacts between two protein structures. It covers the basic usage pattern, explains the example provided in the repository, and demonstrates how to interpret the results. For information about the system architecture, see [System Architecture](/ChengfeiYan/PLMGraph-Inter/2-system-architecture), and for details on the prediction pipeline, see [Prediction Pipeline](/ChengfeiYan/PLMGraph-Inter/4-prediction-pipeline).

## Basic Usage Pattern

PLMGraph-Inter is executed through a command-line interface, with the following general syntax:

```
python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path device
```

Where the parameters are:

| Parameter | Description | Required Format |
| --- | --- | --- |
| sequenceA | FASTA file containing the sequence of protein A | .fasta or .fa file |
| msaA | Multiple sequence alignment for protein A | .a3m file |
| pdbA | Structure file for protein A | .pdb file |
| sequenceB | FASTA file containing the sequence of protein B | .fasta or .fa file |
| msaB | Multiple sequence alignment for protein B | .a3m file |
| pdbB | Structure file for protein B | .pdb file |
| result_path | Directory where results will be stored | Existing directory path |
| device | Computing device to use (CPU or CUDA GPU) | "cpu", "cuda:0", "cuda:1", etc. |

Sources: [README.md L28-L38](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L28-L38)

 [predict.py L40-L41](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L40-L41)

## Example Walkthrough

The repository includes a complete example for the protein complex 1GL1, which consists of two chains: chain A (trypsin) and chain I (trypsin inhibitor).

### Running the Example

To run the provided example, execute:

```
python predict.py ./example/1GL1_A.fasta ./example/1GL1_A_uniref100.a3m ./example/1GL1_A.pdb ./example/1GL1_I.fasta ./example/1GL1_I_uniref100.a3m ./example/1GL1_I.pdb ./example/result cpu
```

This runs the prediction on CPU and stores results in the `./example/result` directory.

Sources: [README.md L40-L41](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L40-L41)

 [example/1GL1_A.fasta L1-L2](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_A.fasta#L1-L2)

 [example/1GL1_I.fasta L1-L2](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/example/1GL1_I.fasta#L1-L2)

### Prediction Flow Diagram

The diagram below illustrates the flow of data when running the example:

```mermaid
flowchart TD

A_fasta["1GL1_A.fasta"]
A_a3m["1GL1_A_uniref100.a3m"]
A_pdb["1GL1_A.pdb"]
B_fasta["1GL1_I.fasta"]
B_a3m["1GL1_I_uniref100.a3m"]
B_pdb["1GL1_I.pdb"]
CLI["Command Line<br>Interface"]
pair_msa["pair_msa.main()"]
tools["External Tools<br>(CCMpred, hhfilter, etc.)"]
feature_extract["Feature Extraction<br>(ESM-1b, MSA-1b, ESM-IF1)"]
graph_build["pdb_graph.main()"]
feature_load["load_feature module"]
model["resnet18 Model"]
pred_file["pred.txt<br>Contact Prediction Matrix"]
other_files["Intermediate Files<br>(MSAs, features, graphs)"]

A_fasta --> CLI
A_a3m --> CLI
A_pdb --> CLI
B_fasta --> CLI
B_a3m --> CLI
B_pdb --> CLI
CLI --> pair_msa
CLI --> feature_extract
CLI --> graph_build
model --> pred_file
pair_msa --> other_files
tools --> other_files
feature_extract --> other_files
graph_build --> other_files

subgraph Output ["Output"]
    pred_file
    other_files
end

subgraph subGraph1 ["Internal Processing"]
    pair_msa
    tools
    feature_extract
    graph_build
    feature_load
    model
    pair_msa --> tools
    tools --> feature_load
    feature_extract --> feature_load
    graph_build --> feature_load
    feature_load --> model
end

subgraph subGraph0 ["Input Files"]
    A_fasta
    A_a3m
    A_pdb
    B_fasta
    B_a3m
    B_pdb
end
```

Sources: [predict.py L59-L62](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L59-L62)

 [predict.py L100-L103](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L100-L103)

 [predict.py L125-L144](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L125-L144)

 [predict.py L153-L155](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L153-L155)

 [predict.py L176-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L176-L201)

## Understanding Inputs and Outputs

### Input Requirements

For successful prediction, you need to prepare the following files for each protein:

1. **FASTA file (.fasta)**: Contains the amino acid sequence of the protein
2. **Multiple Sequence Alignment (.a3m)**: MSA derived from the UniRef100 database
3. **Structure file (.pdb)**: 3D structure coordinates of the protein

The MSA files should be in A3M format and derived from the UniRef100 database. If there are missing residues in the PDB files, you can use MODELLER to fill in these missing residues.

Sources: [README.md L37-L38](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L37-L38)

 [predict.py L40-L41](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L40-L41)

### Output Files and Structure

When you run a prediction, the system generates several types of files:

```mermaid
flowchart TD

paired_files["paired.*<br>(paired.a3m, paired.ccmpred, etc.)"]
A_feature_files["A_*<br>(A_esm1b.repr, A_msa1b.repr, A_esmif.repr)"]
B_feature_files["B_*<br>(B_esm1b.repr, B_msa1b.repr, B_esmif.repr)"]
graph_files["graph*.pkl<br>(graphA.pkl, graphB.pkl)"]
attn_files["*.attn<br>(msa1b_rt.attn, msa1b_sw.attn)"]
pred_txt["pred.txt<br>(Final contact prediction matrix)"]

subgraph result_directory ["result_directory"]
    pred_txt

subgraph subGraph0 ["Intermediate Files"]
    paired_files
    A_feature_files
    B_feature_files
    graph_files
    attn_files
end
end
```

The final prediction result is stored in `pred.txt`, which contains a contact probability matrix. Each value in the matrix (ranging from 0 to 1) represents the predicted probability of contact between residue pairs of protein A and protein B.

Sources: [predict.py L200-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L200-L201)

 [predict.py L49-L103](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L49-L103)

 [predict.py L125-L155](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L125-L155)

## Interpreting Results

The output file `pred.txt` contains a 2D matrix where:

* Rows correspond to residues in protein A
* Columns correspond to residues in protein B
* Values represent the probability (from 0 to 1) of contact between residue pairs

Higher values indicate stronger predicted contacts. A typical visualization of the prediction matrix would look like:

```mermaid
flowchart TD

view["Visual representation<br>of the contact matrix<br>with hotspots indicating<br>likely interaction sites"]
matrix["<br>Prediction Matrix<br>(Example visualization)<br><br>0.01 0.02 0.87 0.03 ...<br>0.02 0.05 0.04 0.91 ...<br>0.75 0.04 0.02 0.04 ...<br>... ... ... ... ...<br>"]
top_contacts["Top-scoring contacts<br>(highest probability values)"]

subgraph subGraph1 ["Interpreting pred.txt"]
    matrix
    top_contacts
    matrix --> top_contacts
    top_contacts --> view

subgraph subGraph0 ["Contact Map Visualization"]
    view
end
end
```

In practice, you would typically:

1. Sort the matrix to find pairs with the highest scores (closest to 1.0)
2. Consider pairs with scores above a certain threshold (e.g., 0.5) as predicted contacts
3. Visualize these contacts on the 3D structures using molecular visualization software

The README shows an example visualization for the 1GL1 complex where brighter regions indicate higher probability contacts.

Sources: [README.md L46-L49](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L46-L49)

 [predict.py L200-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L200-L201)

## Technical Execution Flow

The following diagram illustrates the technical execution flow and how the code components interact:

```mermaid
sequenceDiagram
  participant User
  participant predict.py
  participant paired.pair_msa
  participant pdb_graph.main
  participant plm.*.repr modules
  participant load_feature
  participant model.resnet18

  User->>predict.py: Run with protein files
  predict.py->>paired.pair_msa: Call pair_msa.main(file_dict, 0.5, 100000)
  note over paired.pair_msa: Align MSAs from protein A and B
  paired.pair_msa-->>predict.py: Create paired.a3m
  predict.py->>plm.*.repr modules: Run ESM-1b, MSA-1b, ESM-IF1
  note over plm.*.repr modules: Extract language model features
  plm.*.repr modules-->>predict.py: Create representation files
  predict.py->>pdb_graph.main: Call pdb_graph.main(pdb*, graph*)
  note over pdb_graph.main: Convert PDB to geometric graphs
  pdb_graph.main-->>predict.py: Create graph*.pkl files
  predict.py->>load_feature: Call graph_feature() and paired_feature()
  note over load_feature: Load features into correct format
  load_feature-->>predict.py: Return node/edge features
  predict.py->>model.resnet18: Create model and load weights
  predict.py->>model.resnet18: Forward pass for A→B and B→A
  note over model.resnet18: Calculate contact probabilities
  model.resnet18-->>predict.py: Return prediction matrices
  predict.py->>User: Write averaged predictions to pred.txt
```

Sources: [predict.py L59-L95](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L59-L95)

 [predict.py L107-L144](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L107-L144)

 [predict.py L153-L155](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L153-L155)

 [predict.py L160-L201](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L160-L201)

## Advanced Usage

### Using GPU Acceleration

To use GPU acceleration for faster prediction, specify a CUDA device:

```
python predict.py sequenceA msaA pdbA sequenceB msaB pdbB result_path cuda:0
```

The device parameter can be:

* `cpu` for CPU computation
* `cuda:0` for the first GPU
* `cuda:1` for the second GPU (if available)
* etc.

Sources: [README.md L37](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L37-L37)

 [predict.py L40-L41](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L40-L41)

 [predict.py L171-L172](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L171-L172)

### Understanding the Prediction Process

The PLMGraph-Inter pipeline performs:

1. MSA pairing - combines multiple sequence alignments of the two proteins
2. Feature extraction - computes language model representations from: * ESM-1b (sequence-based features) * ESM-MSA-1b (MSA-based features) * ESM-IF1 (structure-based features)
3. Graph construction - converts protein structures to geometric graphs
4. Contact prediction - using a ResNet model with GVP embeddings

The prediction is the average of 14 separate predictions (7 cross-validation models, each applied in both A-B and B-A directions).

Sources: [predict.py L176-L198](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L176-L198)

 [README.md L55-L57](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L55-L57)

## Troubleshooting and Common Issues

1. **Missing residues in PDB files**: Use MODELLER to fill in missing residues in the input PDB files before running prediction.
2. **MSA quality**: The quality of predictions strongly depends on MSA quality. Ensure MSAs are derived from the UniRef100 database and have sufficient sequence diversity.
3. **Memory requirements**: For large proteins, the system may require substantial memory. Consider running on a machine with at least 16GB RAM.
4. **Path configuration**: Ensure you've properly set the paths of external tools and model weights in [predict.py L25-L35](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L25-L35)

Sources: [README.md L37-L38](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L37-L38)

 [predict.py L25-L35](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L25-L35)

 [README.md L21-L24](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L21-L24)

## Summary of File Organization

This diagram shows the expected file organization when working with PLMGraph-Inter:

```mermaid
flowchart TD

root["PLMGraph-Inter Repository"]
code["Python Code<br>(predict.py, model.py, etc.)"]
data["data/<br>Regression parameters"]
model["model/<br>Trained models"]
examples["example/<br>Example data"]
your_data["your_data/<br>Your protein data"]
proteinA_files["Protein A files<br>(fasta, a3m, pdb)"]
proteinB_files["Protein B files<br>(fasta, a3m, pdb)"]
results["results/<br>Prediction outputs"]

root --> code
root --> data
root --> model
root --> examples
root --> your_data
your_data --> proteinA_files
your_data --> proteinB_files
your_data --> results
```

Sources: [README.md L20-L26](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L20-L26)

 [predict.py L176](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/predict.py#L176-L176)

Remember to:

1. Download all required model weights and place them in the correct location
2. Set the correct paths for external tools in predict.py
3. Create separate directories for each prediction job to avoid file conflicts

For any issues during installation or execution, contact the developers as indicated in the README.

Sources: [README.md L62](https://github.com/ChengfeiYan/PLMGraph-Inter/blob/d1c5ea71/README.md?plain=1#L62-L62)