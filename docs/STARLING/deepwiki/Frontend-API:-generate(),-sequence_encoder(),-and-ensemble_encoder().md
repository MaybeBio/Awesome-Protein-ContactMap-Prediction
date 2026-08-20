# Frontend API: generate\(\), sequence\_encoder\(\), and ensemble\_encoder\(\)

> **Relevant source files**
> - [starling/frontend/ensemble\_generation\.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py)
> - [starling/scripts/starling\_main\_cli\.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/scripts/starling_main_cli.py)
> - [starling/utilities\.py](https://github.com/idptools/starling/blob/4b98d2fe/starling/utilities.py)

 The STARLING frontend API serves as the primary entry point for users to interact with the generative models\. It is designed to handle the complexity of input normalization, validation, and optimization before delegating to the backend generation logic\. This layer ensures that all inputs conform to the physical and architectural constraints of the underlying neural networks\.

## Input Normalization: `handle_input`

 The `handle_input` function is the gatekeeper for all sequence data entering the pipeline\. It supports a wide variety of formats and ensures that sequences are cleaned and validated\.

### Data Flow and Transformation

 `handle_input` transforms various input types into a standardized dictionary format \(`{name: sequence}`\)\.

| Input Type | Handling Logic |
| --- | --- |
| FASTA File | Parsed using protfasta\.read\_fasta\. Supports invalid\_sequence\_action\. starling/frontend/ensemble\_generation\.py85\-90 |
| TSV / \.in File | Parsed as tab\-separated name\\tsequence\. Validates against duplicate names\. starling/frontend/ensemble\_generation\.py91\-104 |
| String | Interpreted as a single amino acid sequence\. starling/frontend/ensemble\_generation\.py106\-120 |
| List | Sequences are indexed automatically \(e\.g\., sequence\_1, sequence\_2\)\. starling/frontend/ensemble\_generation\.py123\-128 |
| Dict | Validated to ensure all values are clean amino acid sequences\. starling/frontend/ensemble\_generation\.py129\-134 |

### Sequence Validation

 The internal `clean_sequence` function enforces the following:

 1. **Standard Residues Only**: Only the 20 canonical amino acids \(`ACDEFGHIKLMNPQRSTVWY`\) are permitted\. [ensemble\_generation\.py L68-L69](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L68-L69)
2. **Case Normalization**: All sequences are converted to uppercase\. [ensemble\_generation\.py L67](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L67-L67)
3. **Strict Length Matching**: If the cleaned sequence length differs from the input length \(due to invalid characters\), a `ValueError` is raised\. [ensemble\_generation\.py L71-L72](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L71-L72)

 **Sources:**

 - `starling/frontend/ensemble_generation.py:10-139]()`

---

## The `generate()` Function

 The `generate()` function is the primary user\-facing API\. It implements a **Validate\-then\-Delegate** pattern: it performs exhaustive sanity checks on parameters and hardware availability before calling `generate_backend`\.

### Parameter Sanity Checks

 `generate()` validates several key parameters to prevent runtime failures in the diffusion loop:

 - **Positive Integers**: `conformations`, `steps`, and `batch_size` must be positive integers\. [ensemble\_generation\.py L228-L251](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L228-L251)
- **Device Selection**: Uses `utilities.check_device` to resolve `None`, `cpu`, `cuda`, or `mps`\. [ensemble\_generation\.py L253-L254](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L253-L254)
- **Sequence Length**: Enforces `configs.MAX_SEQUENCE_LENGTH`\. Any sequence exceeding this limit is flagged, and the function raises a `ValueError`\. [ensemble\_generation\.py L276-L285](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L276-L285)

### Bucketing Optimization

 To maximize GPU throughput, `generate()` groups sequences of the same length into "buckets\." This minimizes the need for excessive padding during the diffusion sampling process\.

 1. Sequences are analyzed and grouped by length\. [ensemble\_generation\.py L287-L293](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L287-L293)
2. Each bucket is processed independently by the backend\. [ensemble\_generation\.py L302-L334](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L302-L334)

### Bridge: Natural Language to Code Space \(Generation\)

 This diagram maps the user's conceptual request to the specific code entities involved in the `generate` workflow\.

```mermaid
flowchart TD

A["'Generate 100 structures for my FASTA file'"]
B["'Use my GPU'"]
C["handle_input()"]
D["check_device()"]
E["generate()"]
F["check_positive_int()"]
G["Sequence Bucketing Logic"]
H["generate_backend()"]

A --> C
B --> D

subgraph subGraph1 ["Code Entity Space (starling/frontend/ensemble_generation.py)"]
    C
    D
    E
    F
    G
    H
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
end

subgraph subGraph0 ["Natural Language Space"]
    A
    B
end
```

 **Sources:**

 - `starling/frontend/ensemble_generation.py:160-334]()`
- `starling/utilities.py:148-241]()`

---

## Sequence and Ensemble Encoders

 Beyond generating distance maps and 3D structures, the frontend provides specialized encoders for sequence\-to\-latent and sequence\-to\-ensemble representations\.

### `sequence_encoder()`

 This function wraps the `SequenceEncoder` model \(a Transformer\-based architecture\)\. It converts raw amino acid sequences into high\-dimensional latent embeddings\.

 - **Input**: Normalized via `handle_input`\. [ensemble\_generation\.py L382](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L382-L382)
- **Validation**: Checks device and sequence length limits\. [ensemble\_generation\.py L385-L397](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L385-L397)
- **Delegation**: Passes the validated dictionary to `sequence_encoder_backend`\. [ensemble\_generation\.py L408](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L408-L408)

### `ensemble_encoder()`

 This function generates a latent representation of the *entire ensemble* for a given sequence\. It essentially runs the diffusion model to produce latents without necessarily decoding them into 3D coordinates\.

 - **Purpose**: Useful for similarity searching or latent space analysis\.
- **Flow**: Similar to `generate()`, but calls `ensemble_encoder_backend`\. [ensemble\_generation\.py L487](https://github.com/idptools/starling/blob/4b98d2fe/starling/frontend/ensemble_generation.py#L487-L487)

### Bridge: Natural Language to Code Space \(Encoding\)

 This diagram illustrates how the encoding functions bridge user strings to model latents\.

```mermaid
flowchart TD

Seq["'MQVTIK...'"]
Func1["sequence_encoder()"]
Func2["ensemble_encoder()"]
BE1["sequence_encoder_backend()"]
BE2["ensemble_encoder_backend()"]

Seq --> Func1
Seq --> Func2
Func1 --> BE1
Func2 --> BE2

subgraph subGraph2 ["Backend Models (starling/inference/generation.py)"]
    BE1
    BE2
end

subgraph subGraph1 ["Frontend API (starling/frontend/ensemble_generation.py)"]
    Func1
    Func2
end

subgraph subGraph0 ["User Input"]
    Seq
end
```

 **Sources:**

 - `starling/frontend/ensemble_generation.py:346-410]()` \(sequence\_encoder\)
- `starling/frontend/ensemble_generation.py:413-489]()` \(ensemble\_encoder\)

---

## Execution Flow Summary

 The following sequence diagram details the interaction between the CLI, the Frontend API, and the Backend\.

```mermaid
sequenceDiagram
  participant starling_main_cli.py
  participant ensemble_generation.py
  participant utilities.py
  participant generation.py

  starling_main_cli.py->>ensemble_generation.py: generate(user_input, conformations, device, ...)
  ensemble_generation.py->>ensemble_generation.py: handle_input(user_input)
  ensemble_generation.py->>utilities.py: check_device(device)
  ensemble_generation.py->>ensemble_generation.py: check_positive_int(conformations)
  note over ensemble_generation.py: Validation: Sequence Length < MAX_SEQUENCE_LENGTH
  ensemble_generation.py->>ensemble_generation.py: Group sequences into length buckets
  loop [for each bucket]
    ensemble_generation.py->>generation.py: generate_backend(sequences, device, ...)
    generation.py-->>ensemble_generation.py: return Ensemble objects
  end
  ensemble_generation.py-->>starling_main_cli.py: return dict[name, Ensemble]
```

 **Sources:**

 - `starling/scripts/starling_main_cli.py:185-200]()`
- `starling/frontend/ensemble_generation.py:276-334]()`
- `starling/utilities.py:148-173]()`
