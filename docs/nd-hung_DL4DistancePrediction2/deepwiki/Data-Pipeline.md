# Data Pipeline

> **Relevant source files**
> * [DataProcessor.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py)
> * [ReadOneProteinFeatures.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadOneProteinFeatures.py)
> * [ReadProteinFeatures.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py)

The data pipeline in this codebase is responsible for transforming raw protein sequence and evolutionary information into structured tensors suitable for training and inference. It handles the transition from per-protein feature files to aggregated `.distanceFeatures.pkl` files, and finally into mini-batches containing 1D sequence features, 2D spatial features, and discretized distance labels.

## Overview of Data Flow

The pipeline follows a multi-stage process to prepare data for the `ResNet4DistMatrix` model. It begins by reading heterogeneous biological features (Secondary Structure, Solvent Accessibility, PSSMs) and evolutionary coupling matrices (CCMpred, PSICOV). These are aggregated and validated before being processed into model-ready tensors through concatenation and dimensionality expansion.

### Natural Language to Code Entity Mapping: Data Flow

The following diagram maps the high-level data stages to the specific functions and scripts that execute them.


**Sources:** [ReadProteinFeatures.py L196-L250](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L196-L250)

 [ReadOneProteinFeatures.py L18-L44](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadOneProteinFeatures.py#L18-L44)

 [DataProcessor.py L109-L205](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L109-L205)

---

## 3.1 Feature Reading & Aggregation

The initial stage involves parsing various bioinformatics tool outputs stored in protein-specific directories. The `ReadProteinFeatures.py` script provides specialized loaders for different feature types, ensuring that all inputs are synchronized with the protein sequence and free of `NaN` values.

Key components include:

* **Sequence-based Features**: Loading 3-state secondary structure (`.ss3`), solvent accessibility (`.acc`), and disorder (`.diso`) [ReadProteinFeatures.py L15-L84](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L15-L84)
* **Evolutionary Profiles**: Integration of PSSM, PSFM, and SS8 features via the `LoadTPLTGT` interface [ReadProteinFeatures.py L227-L236](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L227-L236)
* **Pairwise Matrices**: Loading evolutionary coupling scores from CCMpred and PSICOV, which provide the initial 2D spatial signal [ReadProteinFeatures.py L139-L158](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L139-L158)

For details on file formats and validation logic, see [Feature Reading & Aggregation](/nd-hung/DL4DistancePrediction2/3.1-feature-reading-and-aggregation).

**Sources:** [ReadProteinFeatures.py L10-L13](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L10-L13)

 [ReadProteinFeatures.py L196-L250](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ReadProteinFeatures.py#L196-L250)

---

## 3.2 Feature Processing & Batch Assembly

Once raw features are aggregated into a dictionary, the `DataProcessor.py` module handles the transformation into neural network inputs. This stage is governed by the `modelSpecs` defined in the [Configuration System](/nd-hung/DL4DistancePrediction2/1.2-configuration-system).

Key operations include:

* **Feature Concatenation**: 1D features (like PSSM and SS3) are concatenated into a single sequence feature matrix, while 2D features (like CCMpred and position encodings) are merged into a pairwise feature tensor [DataProcessor.py L137-L205](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L137-L205)
* **Label Discretization**: Continuous atom distances are converted into discrete bins (e.g., `XC` labels) based on cutoffs defined in `config.py` [DataProcessor.py L284-L300](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L284-L300)
* **Batching & Padding**: The pipeline uses a right-alignment padding strategy to handle proteins of varying lengths within a single mini-batch [DataProcessor.py L465-L490](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L465-L490)

For details on tensor construction and weighting strategies, see [Feature Processing & Batch Assembly](/nd-hung/DL4DistancePrediction2/3.2-feature-processing-and-batch-assembly).

**Sources:** [DataProcessor.py L109-L122](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L109-L122)

 [DataProcessor.py L448-L520](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L448-L520)

---

## 3.3 Utility Tensor Operations

A set of specialized mathematical operations in `utils.py` bridges the gap between 1D sequence data and 2D spatial maps. These utilities allow the model to treat sequence-level information as interaction-level information.

### Mapping Utility Functions to Tensor Transformations

This diagram illustrates how specific code entities transform data dimensions.


**Sources:** [utils.py L265-L285](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L265-L285)

 [utils.py L293-L315](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L293-L315)

Key utilities include:

* **OuterConcatenate**: Creates a 2D matrix from a 1D sequence by concatenating feature vectors for every pair $(i, j)$ [utils.py L265-L285](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L265-L285)
* **MidpointFeature**: Extracts features from the sequence midpoint between two residues, providing context for the interaction [utils.py L293-L315](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L293-L315)
* **SampleBoundingBox**: Used during training to crop large proteins into manageable sub-matrices [utils.py L356](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L356-L356)

For details on the geometric logic of these operations, see [Utility Tensor Operations](/nd-hung/DL4DistancePrediction2/3.3-utility-tensor-operations).

**Sources:** [utils.py L265-L315](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L265-L315)

 [DataProcessor.py L12-L15](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DataProcessor.py#L12-L15)