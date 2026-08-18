---
title: "Physical Guidance and Potentials"
source: deepwiki.com
owner: jwohlwend
repo: boltz
url: https://deepwiki.com/jwohlwend/boltz/3.7-physical-guidance-and-potentials
---
# Physical Guidance and Potentials

# Physical Guidance and Potentials

> **Relevant source files**
> - [docs/prediction\.md](https://github.com/jwohlwend/boltz/blob/cb04aecc/docs/prediction.md?plain=1)
> - [src/boltz/data/parse/pdb\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/data/parse/pdb.py)
> - [src/boltz/model/potentials/potentials\.py](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py)

 This document explains the potential system in Boltz that provides physics\-based guidance during structure prediction\. The potential system implements various energy functions that encode chemical and physical constraints, guiding the diffusion process to generate more realistic molecular structures\.

 For information about the core diffusion system that these potentials guide, see [Structure Prediction Module](https://deepwiki.com/jwohlwend/boltz/3.1-boltz-1-model)\. For details about confidence scoring of predicted structures, see [Confidence Prediction](https://deepwiki.com/jwohlwend/boltz/3.2-boltz-2-model)\.

## Overview

 The potential system serves as a physics\-informed guidance mechanism that constrains the diffusion sampling process to produce chemically and physically plausible molecular structures\. It implements various potential energy functions that encode constraints such as bond lengths, angles, chirality, van der Waals interactions, and template\-based structural guidance\.

 **Key Components:**

 - Abstract potential framework for implementing energy functions
- Geometric constraint potentials \(distances, angles, dihedrals\)
- Chemical constraint potentials \(chirality, stereochemistry, planarity\)
- Physical interaction potentials \(van der Waals, contacts\)
- Template\-based guidance potentials
- Time\-dependent parameter scheduling system

 Sources: [potentials\.py L1-L773](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L1-L773)

## System Architecture

 The potential system consists of a hierarchical class structure with abstract base classes defining the interface and specific implementations for different types of constraints\.

  Sources: [potentials\.py L15-L226](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L15-L226) [potentials\.py L231-L772](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L231-L772)

## Potential Framework

 The base `Potential` class defines the core interface that all potential implementations must follow\. Each potential computes energy values and gradients for specific geometric or chemical constraints\.

### Core Methods

| Method | Purpose | Returns |
| --- | --- | --- |
| compute\(\) | Calculate potential energy | Scalar energy value |
| compute\_gradient\(\) | Calculate energy gradient | Gradient tensor |
| compute\_args\(\) | Extract constraint data from features | Index tensors and parameters |
| compute\_variable\(\) | Calculate geometric variable | Distance, angle, or dihedral value |
| compute\_function\(\) | Apply potential function | Energy from geometric variable |

### Computation Flow

  Sources: [potentials\.py L24-L200](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L24-L200) [potentials\.py L213-L225](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L213-L225)

## Potential Types

### Geometric Constraint Potentials

 **DistancePotential**: Computes distance\-based constraints between atom pairs\.

 - Calculates pairwise distances: `r_ij = coords[i] - coords[j]`
- Returns distance norm and gradients
- Used by: `PoseBustersPotential`, `ConnectionsPotential`, `VDWOverlapPotential`

 **DihedralPotential**: Computes dihedral angle constraints for four\-atom sequences\.

 - Calculates dihedral angles using cross products of bond vectors
- Handles chirality and stereochemistry constraints
- Used by: `ChiralAtomPotential`, `StereoBondPotential`, `PlanarBondPotential`

 Sources: [potentials\.py L302-L382](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L302-L382)

### Chemical Constraint Potentials

 **PoseBustersPotential**: Enforces RDKit\-derived geometric constraints including bond lengths, angles, and clash prevention\.

 - Uses `rdkit_bounds_index`, `rdkit_lower_bounds`, `rdkit_upper_bounds` from features
- Applies different buffers for bonds vs angles vs clashes
- Prevents unrealistic molecular geometries

 **ChiralAtomPotential**: Maintains correct chirality at stereogenic centers\.

 - Uses `chiral_atom_index` and `chiral_atom_orientations` from features
- Applies dihedral constraints to preserve R/S stereochemistry
- Buffer parameter: 0\.52360 radians \(30 degrees\)

 **StereoBondPotential**: Enforces E/Z stereochemistry for double bonds\.

 - Uses `stereo_bond_index` and `stereo_bond_orientations` from features
- Applies absolute dihedral constraints for cis/trans geometry
- Buffer parameter: 0\.52360 radians \(30 degrees\)

 Sources: [potentials\.py L384-L581](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L384-L581)

### Physical Interaction Potentials

 **VDWOverlapPotential**: Prevents atomic overlap using van der Waals radii\.

 - Computes pairwise distances between non\-bonded atoms
- Uses `vdw_radii` from constants and `ref_element` features
- Excludes connected chains and single\-atom ions
- Buffer parameter: 22\.5% reduction in VDW radii

 **SymmetricChainCOMPotential**: Enforces symmetry constraints between chain centers of mass\.

 - Uses `symmetric_chain_index` to identify symmetric chain pairs
- Computes center of mass for each chain
- Constrains symmetric chains to have similar distances

 Sources: [potentials\.py L422-L515](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L422-L515)

### Template and Contact Guidance

 **TemplateReferencePotential**: Guides structure prediction using template structures\.

 - Uses `template_cb`, `template_mask_cb`, `template_force` features
- Performs rigid alignment between template and predicted coordinates
- Applies distance constraints based on `template_force_threshold`

 **ContactPotentital**: Incorporates contact prediction guidance\.

 - Uses `contact_pair_index`, `contact_thresholds`, `contact_union_index`
- Supports union operations for alternative contact predictions
- Includes negation masks for anti\-contact constraints

 Sources: [potentials\.py L584-L653](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L584-L653)

## Parameter Scheduling

 The potential system supports time\-dependent parameter scheduling to adjust constraint strength during the diffusion process\.

### Schedule Types

| Schedule Type | Purpose | Parameters |
| --- | --- | --- |
| ExponentialInterpolation | Exponential decay/growth | start, end, alpha |
| PiecewiseStepFunction | Step\-wise changes | thresholds, values |

### Example Parameter Configurations

  Sources: [potentials\.py L202-L211](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L202-L211) [potentials\.py L655-L772](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L655-L772)

## Integration with Diffusion System

 The potential system integrates with the diffusion process through the `get_potentials()` function, which assembles the appropriate set of potentials based on steering configuration\.

### Steering Modes

| Mode | Description | Active Potentials |
| --- | --- | --- |
| fk\_steering | Forward kinematics steering | All geometric and chemical potentials |
| physical\_guidance\_update | Physics\-based guidance | All geometric and chemical potentials |
| contact\_guidance\_update | Contact prediction guidance | All potentials \+ contact/template guidance |

### Guidance vs Resampling

 Each potential supports two operation modes:

 - **Guidance Weight**: Applied during gradient\-based guidance steps
- **Resampling Weight**: Applied during rejection sampling steps
- **Guidance Interval**: Frequency of potential application \(every N steps\)

  Sources: [potentials\.py L655-L772](https://github.com/jwohlwend/boltz/blob/cb04aecc/src/boltz/model/potentials/potentials.py#L655-L772)

---
*Source: [https://deepwiki.com/jwohlwend/boltz/3.7-physical-guidance-and-potentials](https://deepwiki.com/jwohlwend/boltz/3.7-physical-guidance-and-potentials) on DeepWiki*