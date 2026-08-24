# Neural Network Architecture

> **Relevant source files**
> * [DilatedResNet4Distance.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DilatedResNet4Distance.py)
> * [Model4DistancePrediction.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Model4DistancePrediction.py)
> * [ResNet4Distance.py](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ResNet4Distance.py)

This page provides a high-level overview of the deep learning architecture used for protein distance prediction. The system transforms 1D sequence features and 2D evolutionary coupling data into spatial distance maps through a series of specialized layers, including sequence embeddings, 1D/2D convolutions, and residual blocks.

### System Data Flow

The following diagram illustrates the transformation from input sequence data to the final distance prediction matrices, mapping conceptual stages to the specific code entities responsible for them.

**Architecture Overview: Input to Output**


**Sources:** [EmbeddingLayer.py L7-L10](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/EmbeddingLayer.py#L7-L10)

 [Conv1d.py L6-L10](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Conv1d.py#L6-L10)

 [ResNet4Distance.py L6-L12](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ResNet4Distance.py#L6-L12)

 [utils.py L100-L110](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/utils.py#L100-L110)

 [Model4DistancePrediction.py L79-L84](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Model4DistancePrediction.py#L79-L84)

 [NN4LogReg.py L10-L15](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/NN4LogReg.py#L10-L15)

---

### 2.1 Top-Level Model: ResNet4DistMatrix

The `Model4DistancePrediction` module serves as the central orchestrator. It uses a factory-style `BuildModel()` function to instantiate either a standard `ResNet` or a `DilatedResNet` based on configuration flags.

Key responsibilities include:

* **1D-to-2D Transformation**: Converting sequence-length features into square interaction matrices using `OuterConcatenate` or `MidpointFeature`.
* **Multi-Response Handling**: Coordinating multiple prediction heads for different atom pairs (e.g., Cb-Cb, Ca-Ca, N-O).
* **Loss Calculation**: Implementing range-weighted cross-entropy for discrete bins and Gaussian negative log-likelihood for continuous distances.

For details, see [Top-Level Model: ResNet4DistMatrix](/nd-hung/DL4DistancePrediction2/2.1-top-level-model:-resnet4distmatrix).

**Sources:** [Model4DistancePrediction.py L24-L67](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Model4DistancePrediction.py#L24-L67)

 [Model4DistancePrediction.py L383-L420](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Model4DistancePrediction.py#L383-L420)

 [Model4DistancePrediction.py L759](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/Model4DistancePrediction.py#L759-L759)

---

### 2.2 Residual Network Blocks

The core representational power of the model resides in its residual blocks, implemented in `ResNet4Distance.py` and `DilatedResNet4Distance.py`. These layers utilize skip connections to facilitate the training of very deep networks.

* **1D Residuals**: Process sequence-level information before spatial expansion.
* **2D Residuals**: Perform the bulk of the spatial reasoning on the $L \times L$ distance maps.
* **Mask-Aware Normalization**: A custom `batch_norm` implementation that accounts for zero-padding in batches of varying sequence lengths, ensuring that padded positions do not bias the mean and variance calculations.

For details, see [Residual Network Blocks](/nd-hung/DL4DistancePrediction2/2.2-residual-network-blocks).

**Sources:** [ResNet4Distance.py L70-L141](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ResNet4Distance.py#L70-L141)

 [DilatedResNet4Distance.py L79-L156](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/DilatedResNet4Distance.py#L79-L156)

 [ResNet4Distance.py L144-L175](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/ResNet4Distance.py#L144-L175)

---

### 2.3 Prediction Output Layers

The final layers of the network translate the abstract features into interpretable distance distributions. The architecture supports two primary modes:

1. **Discrete Classification (`NN4LogReg`)**: Predicts the probability of the distance falling into specific bins (e.g., 0-4Å, 4-8Å, etc.).
2. **Continuous Estimation (`NN4Normal`)**: Predicts parameters (mean and variance) for a Normal or Bivariate distribution.

**Entity Mapping: Prediction Logic**


For details, see [Prediction Output Layers](/nd-hung/DL4DistancePrediction2/2.3-prediction-output-layers).

**Sources:** [NN4LogReg.py L10-L50](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/NN4LogReg.py#L10-L50)

 [NN4Normal.py L10-L50](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/NN4Normal.py#L10-L50)

 [LogReg.py L6-L25](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/LogReg.py#L6-L25)

---

### 2.4 Embedding Layers

Input sequence identity and categorical features are processed through `EmbeddingLayer.py`. Unlike standard NLP embeddings, these are designed to produce pairwise interaction features.

* **EmbeddingLayer**: Performs tensor products to create $L \times L$ feature maps from 1D embeddings.
* **MetaEmbeddingLayer**: Segregates embeddings by sequence separation (short, medium, and long range).
* **ProfileEmbeddingLayer**: Handles continuous evolutionary profile data using softmax normalization to maintain feature stability.

For details, see [Embedding Layers](/nd-hung/DL4DistancePrediction2/2.4-embedding-layers).

**Sources:** [EmbeddingLayer.py L7-L40](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/EmbeddingLayer.py#L7-L40)

 [EmbeddingLayer.py L165-L167](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/EmbeddingLayer.py#L165-L167)

 [EmbeddingLayer.py L167](https://github.com/nd-hung/DL4DistancePrediction2/blob/11ce7818/EmbeddingLayer.py#L167-L167)