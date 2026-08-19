# Memory Management

> **Relevant source files**
> * [network/memory.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py)

This document covers the memory management utilities and optimization strategies used in RoseTTAFold2. The memory management system provides tools for monitoring tensor memory usage and implementing optimization strategies to handle large protein structures within available memory constraints.

For information about core computational utilities, see [Core Utilities](/uw-ipd/RoseTTAFold2/6.1-core-utilities). For training-specific memory optimizations, see [Training System](/uw-ipd/RoseTTAFold2/5-training-system).

## Memory Reporting System

RoseTTAFold2 includes a dedicated memory reporting utility that provides detailed analysis of tensor memory usage across CPU and GPU devices. The system tracks memory allocation patterns and identifies memory-intensive operations.

```mermaid
flowchart TD

A["mem_report"]
B["gc.get_objects"]
C["Filter torch tensors"]
D["Separate CUDA tensors"]
E["_mem_report('GPU')"]
F["Track data_ptr"]
G["Calculate memory usage"]
H["Print detailed report"]
I["tensor.storage().data_ptr()"]
J["Avoid duplicate counting"]
K["tensor.storage().size()"]
L["Calculate total elements"]
M["tensor.storage().element_size()"]
N["Calculate bytes per element"]
O["Total memory = size × element_size"]

A --> I
E --> K
E --> M

subgraph subGraph1 ["Tensor Analysis"]
    I
    J
    K
    L
    M
    N
    O
    I --> J
    K --> L
    M --> N
    L --> O
    N --> O
end

subgraph subGraph0 ["Memory Reporting Workflow"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
end
```

**Memory Reporting Architecture**

The memory reporting system uses Python's garbage collection to enumerate all tensor objects and provides detailed breakdowns of memory usage patterns.

Sources: [network/memory.py L1-L59](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py#L1-L59)

## Core Memory Management Functions

### Memory Report Function

The `mem_report()` function serves as the main interface for memory analysis:

| Function | Purpose | Key Features |
| --- | --- | --- |
| `mem_report()` | Generate comprehensive memory usage report | GPU/CPU separation, duplicate detection, detailed breakdown |
| `_mem_report(tensors, mem_type)` | Internal reporting for specific device type | Per-tensor analysis, memory threshold filtering |

The system implements several key optimizations:

* **Duplicate Detection**: Uses `tensor.storage().data_ptr()` to avoid counting shared memory blocks multiple times
* **Memory Threshold Filtering**: Only reports tensors consuming more than 128 MB of memory
* **Device-Specific Analysis**: Separates GPU and CPU tensor analysis for targeted optimization

Sources: [network/memory.py L6-L59](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py#L6-L59)

### Memory Calculation Logic

```mermaid
flowchart TD

I["tensor.dtype"]
J["Element type info"]
K["tensor.size()"]
L["Tensor dimensions"]
M["tensor.is_sparse"]
N["Skip sparse tensors"]
O["tensor.is_cuda"]
P["Device classification"]
A["tensor.storage().size()"]
B["Get total elements"]
C["tensor.storage().element_size()"]
D["Get bytes per element"]
E["mem = numel × element_size"]
F["Convert to MB: mem/1024/1024"]
G["Filter > 128 MB tensors"]
H["Accumulate total memory"]

subgraph subGraph1 ["Tensor Properties"]
    I
    J
    K
    L
    M
    N
    O
    P
    I --> J
    K --> L
    M --> N
    O --> P
end

subgraph subGraph0 ["Memory Calculation Process"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    C --> D
    B --> E
    D --> E
    E --> F
    F --> G
    G --> H
end
```

**Memory Calculation Workflow**

Sources: [network/memory.py L23-L47](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py#L23-L47)

## Memory Optimization Strategies

Based on the system architecture, RoseTTAFold2 implements several memory optimization strategies:

```mermaid
flowchart TD

A["Low VRAM Mode"]
B["CPU Offloading"]
C["Gradient Checkpointing"]
D["create_custom_forward"]
E["Mixed Precision"]
F["AMP Training"]
G["Striping Parameters"]
H["get_striping_parameters"]
I["IterativeSimulator"]
J["Memory-aware processing"]
K["Attention Mechanisms"]
L["Memory bottleneck management"]
M["SE3 Transformer"]
N["Geometric processing optimization"]
O["Top-K Neighbors"]
P["Graph construction optimization"]
Q["Sequence Cropping"]
R["subcrop, topk_crop"]
S["Distributed Training"]
T["DDP Multi-GPU"]

B --> I
D --> I
F --> I
H --> I
P --> K
R --> K

subgraph subGraph2 ["Computational Efficiency"]
    O
    P
    Q
    R
    S
    T
    O --> P
    Q --> R
    S --> T
end

subgraph subGraph1 ["Core Processing Integration"]
    I
    J
    K
    L
    M
    N
    I --> J
    K --> L
    M --> N
    J --> K
    J --> M
end

subgraph subGraph0 ["Memory Optimization Components"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    C --> D
    E --> F
    G --> H
end
```

**Memory Optimization Architecture**

## Implementation Details

### Memory Reporting Output Format

The memory reporting system generates structured output with the following format:

```
================================================================================
Element type    Size                    Used MEM(MBytes)
Storage on GPU
-------------------------------------------------------------------------------
[Tensor details for tensors > 128 MB]
-------------------------------------------------------------------------------
Total Tensors: [count]     Used Memory Space: [total MB] MBytes
-------------------------------------------------------------------------------
================================================================================
```

### Key Constants and Thresholds

| Constant | Value | Purpose |
| --- | --- | --- |
| `LEN` | 65 | Report formatting width |
| Memory threshold | 128.0 MB | Minimum size for individual tensor reporting |
| Unit conversion | `/1024/1024` | Bytes to megabytes conversion |

Sources: [network/memory.py L40-L48](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py#L40-L48)

## Integration with Core Systems

The memory management system integrates with the broader RoseTTAFold2 architecture through several key touchpoints:

```mermaid
flowchart TD

A["mem_report()"]
B["Development/Debugging"]
C["Memory Optimization"]
D["IterativeSimulator"]
E["Low VRAM Mode"]
F["Prediction Pipeline"]
G["Gradient Checkpointing"]
H["Training System"]
I["RoseTTAFoldModule"]
J["Neural Network Core"]
K["Attention Mechanisms"]
L["Memory Bottlenecks"]
M["SE3 Transformer"]
N["Geometric Processing"]

D --> I
F --> I
H --> I

subgraph subGraph1 ["System Components"]
    I
    J
    K
    L
    M
    N
    I --> J
    K --> L
    M --> N
    J --> K
    J --> M
end

subgraph subGraph0 ["Memory Management Integration"]
    A
    B
    C
    D
    E
    F
    G
    H
    A --> B
    C --> D
    E --> F
    G --> H
end
```

**Memory Management System Integration**

## Usage Context

The memory management utilities are primarily used during:

1. **Development and Debugging**: Memory profiling to identify bottlenecks
2. **Production Monitoring**: Tracking memory usage patterns in deployed systems
3. **Optimization Validation**: Verifying effectiveness of memory reduction strategies
4. **Hardware Sizing**: Determining appropriate GPU memory requirements

The system focuses specifically on GPU memory management, as this is typically the primary constraint for large protein structure predictions.

Sources: [network/memory.py L1-L59](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/network/memory.py#L1-L59)