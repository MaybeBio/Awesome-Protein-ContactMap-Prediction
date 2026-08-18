# ESM Embeddings

> **Relevant source files**
> * [chai_lab/data/dataset/embeddings/embedding_context.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py)
> * [chai_lab/data/dataset/embeddings/esm.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py)
> * [chai_lab/data/features/generators/esm_generator.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/esm_generator.py)
> * [chai_lab/utils/paths.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/utils/paths.py)

## Purpose and Scope

This document details the ESM (Evolutionary Scale Modeling) embeddings component in the Chai Lab system. ESM embeddings provide rich, contextual representations of protein sequences based on evolutionary information. These embeddings serve as a critical input feature for the structure prediction pipeline, particularly for protein sequences.

For information about other feature generation components like Multiple Sequence Alignments, see [Multiple Sequence Alignments](/chaidiscovery/chai-lab/5.1-multiple-sequence-alignments), or for template-based features, see [Structural Templates](/chaidiscovery/chai-lab/5.2-structural-templates).

## Overview

ESM embeddings in Chai Lab are generated using a pre-trained ESM2 model, specifically the ESM2-3B model with 36 transformer layers. These embeddings capture evolutionary information from protein sequences that helps the diffusion model make accurate structural predictions.

```mermaid
flowchart TD

ProteinSeq["Protein Sequences"]
Tokenizer["DumbTokenizer"]
ESMModel["ESM2 Model<br>3B parameters"]
Embeddings["Per-residue Embeddings"]
EmbeddingContext["EmbeddingContext"]
AllAtomFeatureContext["AllAtomFeatureContext"]
MSAContext["MSAContext"]
TemplateContext["TemplateContext"]
RestraintContext["RestraintContext"]
ModelInference["Structure Prediction Model"]

EmbeddingContext --> AllAtomFeatureContext
AllAtomFeatureContext --> ModelInference

subgraph subGraph1 ["Feature Context Integration"]
    AllAtomFeatureContext
    MSAContext
    TemplateContext
    RestraintContext
    MSAContext --> AllAtomFeatureContext
    TemplateContext --> AllAtomFeatureContext
    RestraintContext --> AllAtomFeatureContext
end

subgraph subGraph0 ["ESM Embedding Generation System"]
    ProteinSeq
    Tokenizer
    ESMModel
    Embeddings
    EmbeddingContext
    ProteinSeq --> Tokenizer
    Tokenizer --> ESMModel
    ESMModel --> Embeddings
    Embeddings --> EmbeddingContext
end
```

Sources: [chai_lab/data/dataset/embeddings/esm.py L140-L180](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L140-L180)

## ESM Model Management

The ESM model is efficiently managed through a global container `_esm_model` and context manager `esm_model()` to minimize memory usage and optimize performance.

### Model Loading and Caching Strategy

The system uses a persistent in-process container `_esm_model` (list) to cache the loaded model and avoid repeated loading: [chai_lab/data/dataset/embeddings/esm.py L16](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L16-L16)

```mermaid
flowchart TD

fork_handler["os.register_at_fork()<br>after_in_child callback"]
clear_cache["_esm_model.clear()"]
global_container["_esm_model: list<br>Global model container"]
context_manager["esm_model(device)<br>Context manager"]
url_constant["ESM_URL<br>Model download URL"]
download_func["download_if_not_exists()"]
cache_folder["esm_cache_folder<br>downloads_path/esm"]
local_path["local_esm_path<br>traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"]
model_inference["Model Inference"]

context_manager --> model_inference

subgraph subGraph0 ["Model Management Components"]
    global_container
    context_manager
    url_constant
    download_func
    cache_folder
    local_path
    global_container --> context_manager
    url_constant --> download_func
    cache_folder --> local_path
    download_func --> local_path
    local_path --> context_manager
end

subgraph subGraph1 ["Process Management"]
    fork_handler
    clear_cache
    fork_handler --> clear_cache
end
```

### Device Management Process

The `esm_model` context manager [chai_lab/data/dataset/embeddings/esm.py L28-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L28-L52)

 handles the lifecycle of the model on the GPU/CPU.

```mermaid
sequenceDiagram
  participant Client Code
  participant esm_model(device)
  participant _esm_model
  participant download_if_not_exists
  participant torch.jit.load

  Client Code->>esm_model(device): with esm_model(device="cuda:0")
  esm_model(device)->>download_if_not_exists: Check/download model
  download_if_not_exists->>esm_model(device): Model file ready
  loop [device != "cuda:0"]
    esm_model(device)->>torch.jit.load: Load model from file
    torch.jit.load->>esm_model(device): Load on CPU first
    esm_model(device)->>esm_model(device): Move to target device
    torch.jit.load->>esm_model(device): Load directly on GPU
    esm_model(device)->>_esm_model: Append model to _esm_model
    esm_model(device)->>_esm_model: Get cached model
  end
  esm_model(device)->>esm_model(device): model.to(device)
  esm_model(device)->>esm_model(device): model.eval()
  esm_model(device)->>Client Code: yield model
  Client Code->>Client Code: Use model for inference
  Client Code->>esm_model(device): Exit context
  esm_model(device)->>esm_model(device): model.to("cpu")
```

The system also registers a fork handler using `os.register_at_fork(after_in_child=lambda: _esm_model.clear())` to prevent model sharing issues in multi-process environments [chai_lab/data/dataset/embeddings/esm.py L18](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L18-L18)

Sources: [chai_lab/data/dataset/embeddings/esm.py L16-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L16-L52)

 [chai_lab/utils/paths.py L29-L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/utils/paths.py#L29-L48)

## Tokenization Process

Before a protein sequence can be processed by the ESM model, it needs to be tokenized. Chai Lab implements a `DumbTokenizer` that converts amino acid sequences to token IDs [chai_lab/data/dataset/embeddings/esm.py L91-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L91-L109)

### Token Mapping and DumbTokenizer Implementation

The system defines a comprehensive `token_map` dictionary that maps amino acids and special tokens to integer IDs [chai_lab/data/dataset/embeddings/esm.py L54-L88](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L54-L88)

```mermaid
flowchart TD

raw_seq["Raw Protein Sequence<br>'MKTLVLGAVILGSTLLAG...'"]
add_special["Add BOS/EOS<br>f'<cls>{seq}<eos>'"]
tokenize_call["esm_tokenizer.tokenize()"]
tensor_convert["torch.asarray()<br>token_ids.to(device)"]
model_input["ESM Model Input<br>tokens=[0, 20, 15, 11, 4, 7, 4, 6...]"]
tokenizer_init["DumbTokenizer.init<br>(token_map)"]
tokenize_method["DumbTokenizer.tokenize<br>(text: str) -> list[int]"]
greedy_match["Greedy Token Matching<br>text.startswith(token, i)"]
token_map["token_map<br>Dict[str, int]"]
special_tokens["Special Tokens<br><cls>:0, <pad>:1, <eos>:2, <unk>:3<br><mask>:32"]
amino_acids["Amino Acids<br>L:4, A:5, G:6, V:7, S:8, E:9<br>R:10, T:11, I:12, D:13, P:14<br>K:15, Q:16, N:17, F:18, Y:19<br>M:20, H:21, W:22, C:23"]
extended_aa["Extended AA<br>X:24, B:25, U:26, Z:27, O:28"]
gap_tokens["Gap Tokens<br>.:29, -:30"]

subgraph subGraph2 ["ESM Processing Pipeline"]
    raw_seq
    add_special
    tokenize_call
    tensor_convert
    model_input
    raw_seq --> add_special
    add_special --> tokenize_call
    tokenize_call --> tensor_convert
    tensor_convert --> model_input
end

subgraph subGraph1 ["DumbTokenizer Class"]
    tokenizer_init
    tokenize_method
    greedy_match
    tokenizer_init --> tokenize_method
    tokenize_method --> greedy_match
end

subgraph subGraph0 ["Token Mapping Components"]
    token_map
    special_tokens
    amino_acids
    extended_aa
    gap_tokens
    token_map --> special_tokens
    token_map --> amino_acids
    token_map --> extended_aa
    token_map --> gap_tokens
end
```

Sources: [chai_lab/data/dataset/embeddings/esm.py L54-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L54-L109)

## Embedding Generation Process

The main function for generating embeddings is `get_esm_embedding_context`, which processes a list of `Chain` objects to generate embeddings [chai_lab/data/dataset/embeddings/esm.py L141-L180](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L141-L180)

### Core Function Flow

```mermaid
flowchart TD

entity_check["chain.entity_data.entity_type<br>== EntityType.PROTEIN"]
get_protein_emb["chain_embs.append<br>(protein_seq2emb_context[sequence])"]
get_zero_emb["chain_embs.append<br>(EmbeddingContext.empty(n_tokens))"]
input_chains["Input: list[Chain]<br>chains parameter"]
extract_seqs["Extract Unique Protein Sequences<br>chain.entity_data.entity_type == EntityType.PROTEIN"]
call_helper["_get_esm_contexts_for_sequences<br>(prot_sequences: set[str], device)"]
process_chains["Process Each Chain<br>for chain in chains"]
explode_embs["Map to Structure Tokens<br>embedding.esm_embeddings[chain.structure_context.token_residue_index, :]"]
merge_embs["torch.cat(cropped_embs, dim=0)<br>Merge along tokens dimension"]
final_output["Return EmbeddingContext<br>(esm_embeddings=merged_embs)"]
empty_check["len(prot_sequences) == 0"]
return_empty["return {}"]
init_dict["seq2embedding_context = {}"]
torch_no_grad["torch.no_grad()"]
esm_context["with esm_model(device=device)"]
seq_loop["for seq in prot_sequences"]
tokenize_seq["esm_tokenizer.tokenize<br>(f'<cls>{seq}<eos>')"]
model_call["model(tokens=token_ids)"]
extract_hidden["last_hidden_state[0, 1:-1]<br>Remove BOS/EOS, move to CPU"]
create_emb_ctx["EmbeddingContext<br>(esm_embeddings=esm_embeddings)"]
store_result["seq2embedding_context[seq] = context"]

subgraph subGraph2 ["get_esm_embedding_context Function"]
    input_chains
    extract_seqs
    call_helper
    process_chains
    explode_embs
    merge_embs
    final_output
    input_chains --> extract_seqs
    extract_seqs --> call_helper
    call_helper --> process_chains
    process_chains --> explode_embs
    explode_embs --> merge_embs
    merge_embs --> final_output

subgraph subGraph1 ["Chain Processing Logic"]
    entity_check
    get_protein_emb
    get_zero_emb
    entity_check --> get_protein_emb
    entity_check --> get_zero_emb
end

subgraph subGraph0 ["_get_esm_contexts_for_sequences Implementation"]
    empty_check
    return_empty
    init_dict
    torch_no_grad
    esm_context
    seq_loop
    tokenize_seq
    model_call
    extract_hidden
    create_emb_ctx
    store_result
    empty_check --> return_empty
    empty_check --> init_dict
    init_dict --> torch_no_grad
    torch_no_grad --> esm_context
    esm_context --> seq_loop
    seq_loop --> tokenize_seq
    tokenize_seq --> model_call
    model_call --> extract_hidden
    extract_hidden --> create_emb_ctx
    create_emb_ctx --> store_result
end
end
```

### Entity Type Handling

The system handles different molecular entity types by checking `chain.entity_data.entity_type` [chai_lab/data/dataset/embeddings/esm.py L155-L161](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L155-L161)

:

| Entity Type | Embedding Strategy | Implementation |
| --- | --- | --- |
| `EntityType.PROTEIN` | ESM model embeddings | `protein_seq2emb_context[chain.entity_data.sequence]` |
| Non-protein entities | Zero embeddings | `EmbeddingContext.empty(n_tokens=chain.structure_context.num_tokens)` |

Sources: [chai_lab/data/dataset/embeddings/esm.py L112-L180](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L112-L180)

 [chai_lab/data/dataset/embeddings/embedding_context.py L48-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L48-L52)

## Key Data Structures

The primary data structure is the `EmbeddingContext`, which contains the ESM embeddings for all tokens in a structure.

### EmbeddingContext

Defined in [chai_lab/data/dataset/embeddings/embedding_context.py L15-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L15-L52)

 this dataclass stores the `esm_embeddings` tensor with shape `[num_tokens, d_emb]`. It provides utility methods like `pad` [chai_lab/data/dataset/embeddings/embedding_context.py L28-L42](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L28-L42)

 and `empty` [chai_lab/data/dataset/embeddings/embedding_context.py L48-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L48-L52)

### ESMEmbeddings Feature Generator

The `ESMEmbeddings` class [chai_lab/data/features/generators/esm_generator.py L12-L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/esm_generator.py#L12-L35)

 is a `FeatureGenerator` that extracts embeddings from a batch and prepares them for the model trunk.

```mermaid
classDiagram
    class EmbeddingContext {
        +Tensor esm_embeddings
        +num_tokens int
        +pad(max_tokens) : EmbeddingContext
        +empty(n_tokens, d_emb) : static
    }
    class ESMEmbeddings {
        +FeatureType ty
        +get_input_kwargs_from_batch(batch)
        +_generate(esm_embeddings)
    }
    ESMEmbeddings ..> EmbeddingContext : processes
```

Sources: [chai_lab/data/dataset/embeddings/embedding_context.py L15-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L15-L52)

 [chai_lab/data/features/generators/esm_generator.py L12-L35](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/features/generators/esm_generator.py#L12-L35)

## Implementation Details

### Model Loading and Device Strategy

The system implements optimized loading based on target device [chai_lab/data/dataset/embeddings/esm.py L38-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L38-L43)

:

```mermaid
flowchart TD

check_device["device != torch.device('cuda:0')"]
load_cpu["torch.jit.load(local_esm_path,<br>map_location='cpu')<br>.to(device)"]
load_gpu["torch.jit.load(local_esm_path)<br>.to(device)"]
append_model["_esm_model.append(model)"]

subgraph subGraph0 ["Model Loading Strategy"]
    check_device
    load_cpu
    load_gpu
    append_model
    check_device --> load_cpu
    check_device --> load_gpu
    load_cpu --> append_model
    load_gpu --> append_model
end
```

The model is moved back to CPU using `model.to("cpu")` after the context manager yields to prevent GPU memory bloat [chai_lab/data/dataset/embeddings/esm.py L51](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L51-L51)

### Embedding Dimensionality

By default, the `EmbeddingContext.empty` method uses a dimension of `2560` [chai_lab/data/dataset/embeddings/embedding_context.py L48](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L48-L48)

 which corresponds to the ESM2-3B output dimension.

Sources: [chai_lab/data/dataset/embeddings/esm.py L28-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L28-L52)

 [chai_lab/data/dataset/embeddings/embedding_context.py L48-L52](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/embedding_context.py#L48-L52)

## Technical Specifications

The ESM model used in Chai Lab is:

| Specification | Value |
| --- | --- |
| Model | ESM2-t36-3B |
| Size | 3 billion parameters |
| Architecture | 36-layer transformer |
| Format | Traced model (PyTorch JIT) |
| Precision | FP16 (half precision) |
| Download URL | `ESM_URL` [chai_lab/data/dataset/embeddings/esm.py L21](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L21-L21) |
| Local Path | `downloads_path / "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"` |

### File Path Management

```mermaid
flowchart TD

esm_url["ESM_URL<br>'Unsupported markdown: link'"]
download_func["download_if_not_exists()<br>Download if missing"]
cache_folder["esm_cache_folder<br>downloads_path.joinpath('esm')"]
local_path["local_esm_path<br>downloads_path.joinpath('esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt')"]

subgraph subGraph0 ["ESM Model Path Management"]
    esm_url
    download_func
    cache_folder
    local_path
    esm_url --> download_func
    cache_folder --> local_path
    download_func --> local_path
end
```

Sources: [chai_lab/data/dataset/embeddings/esm.py L21-L34](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/embeddings/esm.py#L21-L34)

 [chai_lab/utils/paths.py L19-L22](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/utils/paths.py#L19-L22)