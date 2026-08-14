# Structure Representation

> **Relevant source files**
> * [src/alphafold3/structure/__init__.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/__init__.py)
> * [src/alphafold3/structure/bonds.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py)
> * [src/alphafold3/structure/chemical_components.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py)
> * [src/alphafold3/structure/cpp/string_array.pyi](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/cpp/string_array.pyi)
> * [src/alphafold3/structure/structure.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py)
> * [src/alphafold3/structure/structure_tables.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py)
> * [src/alphafold3/structure/table.py](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py)

This page documents the `Structure` class and its table-based architecture for representing molecular structures in AlphaFold 3. The `Structure` class is the primary container for storing atomic coordinates, chain information, residues, and bonds in a normalized, database-like format that enables efficient filtering, iteration, and manipulation of structural data.

## Core Architecture

The `Structure` class implements a table-based design inspired by relational databases, where molecular data is organized into four interconnected tables: chains, residues, atoms, and bonds [src/alphafold3/structure/structure.py L296-L344](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L296-L344)

 This architecture provides several advantages:

* **Normalized storage**: Avoids data duplication by storing chain and residue metadata once.
* **Efficient filtering**: Operations can cascade through foreign key relationships [src/alphafold3/structure/structure.py L301-L305](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L301-L305)
* **Type safety**: Each table is a strongly-typed dataclass with validated columns [src/alphafold3/structure/table.py L49-L63](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py#L49-L63)
* **Flexible indexing**: Supports both flat (atom-level) and hierarchical (chain/residue) access patterns.

**Table-Based Structure Architecture**

```

```

Sources: [src/alphafold3/structure/structure.py L296-L344](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L296-L344)

 [src/alphafold3/structure/structure.py L270-L275](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L270-L275)

 [src/alphafold3/structure/structure_tables.py L78-L89](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L78-L89)

 [src/alphafold3/structure/structure_tables.py L203-L210](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L203-L210)

 [src/alphafold3/structure/structure_tables.py L218-L228](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L218-L228)

### Foreign Key Relationships

The `Structure` class enforces referential integrity through foreign keys defined at the class level [src/alphafold3/structure/structure.py L301-L305](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L301-L305)

 These relationships are validated during construction to ensure all referenced keys exist in their target tables [src/alphafold3/structure/structure.py L345-L381](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L345-L381)

```

```

Sources: [src/alphafold3/structure/structure.py L301-L305](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L301-L305)

 [src/alphafold3/structure/structure.py L345-L381](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L345-L381)

## Data Tables

The tables are implemented as subclasses of the `Table` base class [src/alphafold3/structure/table.py L49-L63](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py#L49-L63)

 which provides core functionality for indexing by key [src/alphafold3/structure/table.py L91-L102](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py#L91-L102)

 slicing by boolean mask [src/alphafold3/structure/table.py L127-L135](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py#L127-L135)

 and accessing columns as numpy arrays [src/alphafold3/structure/table.py L110-L112](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/table.py#L110-L112)

### Chains Table

The `Chains` table stores metadata for each molecular chain [src/alphafold3/structure/structure_tables.py L218-L228](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L218-L228)

| Column | Type | Description |
| --- | --- | --- |
| `key` | int64 | Unique chain identifier (internal) |
| `id` | object | Label chain ID (mmCIF `label_asym_id`) |
| `type` | object | Chain type (e.g., `"polypeptide(L)"`) |
| `auth_asym_id` | object | Author chain ID (mmCIF `auth_asym_id`) |
| `entity_id` | object | Entity identifier |
| `entity_desc` | object | Entity description |

Sources: [src/alphafold3/structure/structure_tables.py L218-L228](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L218-L228)

 [src/alphafold3/structure/structure.py L89-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L89-L96)

### Residues Table

The `Residues` table stores metadata for each residue (resolved or unresolved) [src/alphafold3/structure/structure_tables.py L203-L210](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L203-L210)

| Column | Type | Description |
| --- | --- | --- |
| `key` | int64 | Unique residue identifier (internal) |
| `chain_key` | int64 | Foreign key to chains table |
| `id` | int32 | Label residue ID (mmCIF `label_seq_id`) |
| `name` | object | Residue name (CCD code, e.g., `"ALA"`) |
| `auth_seq_id` | object | Author residue ID (mmCIF `auth_seq_id`) |
| `insertion_code` | object | PDB insertion code |

Sources: [src/alphafold3/structure/structure_tables.py L203-L210](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L203-L210)

 [src/alphafold3/structure/structure.py L98-L104](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L98-L104)

### Atoms Table

The `Atoms` table stores atomic coordinates and properties. It supports multi-model data through leading dimensions in coordinate columns [src/alphafold3/structure/structure_tables.py L90-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L90-L96)

| Column | Type | Description |
| --- | --- | --- |
| `key` | int64 | Unique atom identifier (internal) |
| `chain_key` | int64 | Foreign key to chains table |
| `res_key` | int64 | Foreign key to residues table |
| `name` | object | Atom name (e.g., `"CA"`) |
| `element` | object | Element symbol |
| `x`, `y`, `z` | float32 | Cartesian coordinates |
| `b_factor` | float32 | Temperature factor |
| `occupancy` | float32 | Occupancy value |

Sources: [src/alphafold3/structure/structure_tables.py L78-L96](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L78-L96)

 [src/alphafold3/structure/structure.py L106-L115](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L106-L115)

### Bonds Table

The `Bonds` table stores connectivity information, typically parsed from `_struct_conn` categories [src/alphafold3/structure/bonds.py L23-L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py#L23-L42)

| Column | Type | Description |
| --- | --- | --- |
| `key` | int64 | Unique bond identifier (internal) |
| `type` | object | Bond type (e.g., `"covale"`, `"disulf"`) |
| `role` | object | Bond role (e.g., `"N-Glycosylation"`) |
| `from_atom_key` | int64 | Foreign key to atoms table |
| `dest_atom_key` | int64 | Foreign key to atoms table |

Sources: [src/alphafold3/structure/bonds.py L23-L42](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/bonds.py#L23-L42)

 [src/alphafold3/structure/structure.py L412-L414](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L412-L414)

## Construction Methods

**Structure Construction Pathways**

```

```

Sources: [src/alphafold3/structure/parsing.py L399-L453](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L399-L453)

 [src/alphafold3/structure/parsing.py L456-L588](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L456-L588)

 [src/alphafold3/structure/parsing.py L628-L827](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L628-L827)

### mmCIF Parsing

The `from_mmcif()` and `from_parsed_mmcif()` functions handle the conversion of mmCIF data. The process involves resolving alternate locations (alt-locs) and applying entity filters using C++ utilities [src/alphafold3/structure/parsing.py L307-L396](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L307-L396)

* **Alt-loc Resolution**: Handled in C++ to select the highest occupancy conformer [src/alphafold3/structure/parsing.py L307-L396](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L307-L396)
* **Bonds Parsing**: Extracted from `_struct_conn` mmCIF categories [src/alphafold3/structure/parsing.py L202-L253](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L202-L253)

### from_res_arrays()

This method constructs a `Structure` from residue-centric arrays, typically used when converting model predictions back into the internal `Structure` representation [src/alphafold3/structure/parsing.py L456-L588](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L456-L588)

### from_sequences_and_bonds()

Creates a structure from sequence strings and explicit bond definitions. It uses the Chemical Components Dictionary (CCD) to lookup standard atom names and elements [src/alphafold3/structure/parsing.py L628-L827](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/parsing.py#L628-L827)

## Data Access and Filtering

### Iteration and Broadcasting

The `Structure` class provides properties that broadcast table data to the atom level using foreign key joins [src/alphafold3/structure/structure.py L524-L585](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L524-L585)

* `struc.atom_name`: Array of names for every atom.
* `struc.chain_id`: Array of chain IDs broadcasted to every atom.
* `struc.res_name`: Array of residue names broadcasted to every atom.

### Filtering Operations

The `filter()` method creates a new `Structure` by applying predicates to columns [src/alphafold3/structure/structure.py L1076-L1165](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1076-L1165)

```

```

Filtering is "cascade-aware". When atoms are removed, the `Structure` class can optionally delete residues or chains that no longer contain any atoms via the `CascadeDelete` enum [src/alphafold3/structure/structure.py L42-L46](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L42-L46)

 [src/alphafold3/structure/structure.py L1167-L1240](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1167-L1240)

### Entity Type Filtering

Convenience methods exist to filter by biological type [src/alphafold3/structure/structure.py L1319-L1388](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1319-L1388)

:

* `filter_to_entity_type(protein=True)`: Keeps only polypeptide chains.
* `filter_to_entity_type(ligand=True)`: Keeps non-polymer/branched entities.

## Coordinate Representations

### Flat Atom Arrays

Coordinates are stored as `(num_atoms, 3)` arrays in `struc.coords` [src/alphafold3/structure/structure.py L1551-L1562](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1551-L1562)

 For multi-model structures, the leading dimension represents the model index [src/alphafold3/structure/structure_tables.py L187-L200](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L187-L200)

### Residue Arrays (ResArrays)

The `to_res_arrays()` method converts the structure into a fixed-width residue representation [src/alphafold3/structure/structure.py L1389-L1493](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1389-L1493)

 This is critical for model featurization.

* **Atom Ordering**: A mapping defines the position of each atom type in the resulting array.
* **Masking**: A boolean `atom_mask` indicates which atoms are physically present in the structure vs. padded [src/alphafold3/structure/structure.py L278-L293](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L278-L293)

Sources: [src/alphafold3/structure/structure.py L1389-L1493](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1389-L1493)

 [src/alphafold3/structure/structure.py L1551-L1562](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L1551-L1562)

 [src/alphafold3/structure/structure_tables.py L187-L200](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure_tables.py#L187-L200)

## Chemical Components Integration

The `Structure` class maintains a `ChemicalComponentsData` object [src/alphafold3/structure/chemical_components.py L71-L79](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L71-L79)

 which provides metadata for components like SMILES strings [src/alphafold3/structure/chemical_components.py L38](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L38-L38)

 synonyms [src/alphafold3/structure/chemical_components.py L34](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L34-L34)

 and RDKit molecule objects [src/alphafold3/structure/chemical_components.py L56-L61](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L56-L61)

**Chemical Data Mapping**

```

```

Sources: [src/alphafold3/structure/chemical_components.py L25-L61](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L25-L61)

 [src/alphafold3/structure/chemical_components.py L71-L79](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/chemical_components.py#L71-L79)

 [src/alphafold3/structure/structure.py L317-L318](https://github.com/google-deepmind/alphafold3/blob/97639fff/src/alphafold3/structure/structure.py#L317-L318)