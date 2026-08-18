---
title: "Glycan Processing"
source: deepwiki.com
owner: chaidiscovery
repo: chai-lab
url: https://deepwiki.com/chaidiscovery/chai-lab/4.4-glycan-processing
---
# Glycan Processing

# Glycan Processing

> **Relevant source files**
> - [chai\_lab/data/dataset/structure/all\_atom\_residue\_tokenizer\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py)
> - [chai\_lab/data/parsing/glycans\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py)
> - [tests/test\_glycans\.py](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py)

## Purpose and Scope

 The glycan processing system handles the parsing and representation of glycan \(carbohydrate\) structures from string\-based input into internal data structures used by the Chai\-1 model\. This system is specifically designed to process glycan entities identified during input parsing and convert them into the unified `AllAtomStructureContext` representation\.

 For information about other molecular entity types, see [Entity Type Identification](https://deepwiki.com/chaidiscovery/chai-lab/4.2-entity-type-identification)\. For broader input processing workflows, see [Input Processing](https://deepwiki.com/chaidiscovery/chai-lab/4-input-processing)\.

## Glycan String Format

 The system processes glycan structures using a specialized string notation that represents the hierarchical branching structure of carbohydrates\. This format uses three\-letter Chemical Component Dictionary \(CCD\) codes for individual sugars and parenthetical notation for branch structures\.

### String Structure Components

| Component | Description | Example |
| --- | --- | --- |
| Sugar Code | Three\-letter CCD code | MAN, 99K, FUC |
| Bond Notation | src\-dst format \(1\-6 indexed\) | 4\-1, 6\-1 |
| Branching | Parentheses for nested structures | \(6\-1 FUC\) |

### Example Glycan Strings

 - **Simple**: `MAN` [test\_glycans\.py L14-L19](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L14-L19)
- **Linear**: `NAG(4-1 NAG)` [test\_glycans\.py L70](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L70-L70)
- **Branched**: `MAN(6-1 FUC)(4-1 MAN)` [glycans\.py L11](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L11-L11)
- **Complex**: `MAN(4-1 FUC(4-1 MAN)(6-1 FUC(4-1 MAN)))(6-1 MAN(6-1 MAN(4-1 MAN)(6-1 FUC)))` [test\_glycans\.py L46-L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L46-L49)

 Sources: [glycans\.py L10-L12](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L10-L12) [test\_glycans\.py L22-L49](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L22-L49)

## Parsing Pipeline

 The glycan processing pipeline converts string representations into structured data through a multi\-step parsing process\.

### Code\-to\-Entity Mapping

 The following diagram maps the natural language concept of glycan parsing to the specific code entities in the `chai-lab` repository\.

 **Diagram: Glycan Parsing Data Flow**

  Sources: [glycans\.py L46-L111](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L46-L111) [all\_atom\_residue\_tokenizer\.py L17-L26](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py#L17-L26)

### Core Parsing Logic

 The `_glycan_string_to_sugars_and_bonds` function performs the primary parsing logic:

 1. **Character\-by\-character parsing**: Processes the glycan string sequentially [glycans\.py L56-L88](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L56-L88)
2. **Bracket tracking**: Maintains parent\-child relationships using a stack \(`parent_sugar_idx`\) to handle branching [glycans\.py L61-L69](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L61-L69)
3. **Bond extraction**: Identifies bond patterns \(`[1-6]-[1-6]`\) using regex and creates `GlycosidicBond` objects [glycans\.py L72-L82](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L72-L82)
4. **Sugar identification**: Extracts three\-letter CCD codes for individual sugars [glycans\.py L84-L87](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L84-L87)

 Sources: [glycans\.py L46-L91](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L46-L91)

## Data Structures

### GlycosidicBond

 The `GlycosidicBond` dataclass represents covalent bonds between sugar residues:

  Key properties:

 - **O\-glycosidic bonds**: Links between sugars are modeled as O\-glycosidic bonds; the system uses `src O` and `dst C` [glycans\.py L36-L42](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L36-L42)
- **Atom naming**: `src_atom_name` returns `O{src_atom}`, `dst_atom_name` returns `C{dst_atom}` [glycans\.py L37-L42](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L37-L42)
- **Validation**: The `__post_init__` method ensures that `src_sugar_index` and `dst_sugar_index` are distinct and that atom indices are positive [glycans\.py L30-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L30-L32)

 Sources: [glycans\.py L23-L43](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L23-L43)

### Residue Conversion

 The `glycan_string_residues` function converts parsed sugars into `Residue` objects [glycans\.py L94-L110](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L94-L110)

| Residue Property | Value | Description |
| --- | --- | --- |
| name | Sugar CCD code | Three\-letter identifier \(e\.g\., "MAN"\) |
| label\_seq | i \+ 1 | Sequential \(1\-based\) labels |
| restype | rc\.residue\_types\_with\_nucleotides\_order\["X"\] | Mapped to generic residue type "X" |
| is\_covalent\_bonded | True | Indicates covalent connectivity for glycans |

 Sources: [glycans\.py L94-L111](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L94-L111)

## Integration with Tokenizer and Context

 The glycan processing system integrates with the broader structure prediction pipeline through the `AllAtomResidueTokenizer` and `AllAtomStructureContext`\.

 **Diagram: Glycan Atom Representation**

  Sources: [all\_atom\_residue\_tokenizer\.py L145-L177](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/dataset/structure/all_atom_residue_tokenizer.py#L145-L177) [test\_glycans\.py L87-L108](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L87-L108)

### Atom Management and Bond Effects

 When glycosidic bonds form between sugars, the system:

 1. **Atom existence masks**: Tracks which atoms are present\. A bond between two NAG components \(C8 H15 N O6\) typically displaces one oxygen atom\. For a NAG\(4\-1 NAG\) dimer, the system expects 29 heavy atoms \(15 \+ 15 \- 1\) [test\_glycans\.py L85-L89](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L85-L89)
2. **Covalent bond indices**: Maintains explicit O\-C bond connections between sugars [test\_glycans\.py L99-L108](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L99-L108)
3. **Element tracking**: Preserves atomic composition\. In a NAG dimer, this results in 16 Carbons \(element 6\), 2 Nitrogens \(element 7\), and 11 Oxygens \(element 8\) [test\_glycans\.py L90-L97](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L90-L97)

 Sources: [test\_glycans\.py L68-L109](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/tests/test_glycans.py#L68-L109)

## Error Handling

 The parsing system includes validation for:

 - **Bracket matching**: Ensures `open_count == closed_count` [glycans\.py L90](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L90-L90)
- **Valid patterns**: Raises `ValueError` if a chunk is neither a valid bond pattern \(`[1-6]-[1-6]`\) nor a 3\-letter CCD code \(`[0-9A-Z]{3}`\) [glycans\.py L88-L89](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L88-L89)
- **Non\-empty results**: Raises `ValueError` if no residues are parsed from the string [glycans\.py L96-L97](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L96-L97)

 Sources: [glycans\.py L30-L32](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L30-L32) [glycans\.py L88-L90](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L88-L90) [glycans\.py L96-L97](https://github.com/chaidiscovery/chai-lab/blob/66c38d1f/chai_lab/data/parsing/glycans.py#L96-L97)

---
*Source: [https://deepwiki.com/chaidiscovery/chai-lab/4.4-glycan-processing](https://deepwiki.com/chaidiscovery/chai-lab/4.4-glycan-processing) on DeepWiki*