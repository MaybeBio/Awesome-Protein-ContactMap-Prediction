---
title: "PDBx/mmCIF Processing"
source: deepwiki.com
owner: uw-ipd
repo: RoseTTAFold2
url: https://deepwiki.com/uw-ipd/RoseTTAFold2/7.1-pdbxmmcif-processing
---
# PDBx/mmCIF Processing

# PDBx/mmCIF Processing

> **Relevant source files**
> - [input\_prep/pdbx/reader/PdbxContainers\.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py)
> - [input\_prep/pdbx/reader/PdbxParser\.py](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py)

 This document covers the PDBx/mmCIF file format processing system used in RoseTTAFold2 for parsing and handling crystallographic information files\. The system provides comprehensive support for reading, writing, and manipulating mmCIF format files that contain protein structure data and metadata\.

 For structure analysis and comparison tools, see [Structure Analysis Tools](https://deepwiki.com/uw-ipd/RoseTTAFold2/7.2-structure-analysis-tools)\. For general input processing workflows, see [Input Processing](https://deepwiki.com/uw-ipd/RoseTTAFold2/4.2-input-processing)\.

## Overview

 The PDBx/mmCIF processing system consists of a collection of container classes and parser utilities that handle the Complex Information File \(CIF\) format used by the Protein Data Bank\. This system enables RoseTTAFold2 to process template structures and structural metadata from crystallographic databases\.

 The mmCIF format organizes data in a hierarchical structure with data blocks containing categories, which in turn contain attributes and values\. This mirrors a relational database model where categories are analogous to tables and attributes to columns\.

## Core Architecture

 The PDBx/mmCIF processing system follows a layered architecture with distinct responsibilities for parsing, data representation, and output formatting\.

  **Sources:** [PdbxContainers\.py L1-L820](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L1-L820) [PdbxParser\.py L1-L639](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L1-L639)

## Container Classes

 The container system provides a hierarchical data model for representing mmCIF content in memory\.

### Base Container Architecture

  **Sources:** [PdbxContainers\.py L77-L172](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L77-L172) [PdbxContainers\.py L174-L208](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L174-L208) [PdbxContainers\.py L210-L229](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L210-L229)

### Key Container Classes

| Class | Purpose | Key Methods |
| --- | --- | --- |
| ContainerBase | Base class for all containers | append\(\), getObj\(\), exists\(\) |
| DataContainer | Holds data blocks | setGlobal\(\), getGlobal\(\) |
| DefinitionContainer | Holds definition sections | isCategory\(\), isAttribute\(\) |
| DataCategory | Stores category data | getValue\(\), setValue\(\), appendAttribute\(\) |
| CifName | Utility for CIF name parsing | categoryPart\(\), attributePart\(\) |

 The `DataCategory` class serves as the primary data storage mechanism, implementing list\-like behavior with support for both integer indexing \(for rows\) and string indexing \(for attribute access\)\.

 **Sources:** [PdbxContainers\.py L48-L75](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L48-L75) [PdbxContainers\.py L275-L820](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L275-L820)

## Parser Implementation

 The parsing system uses a state machine approach with regex\-based tokenization to process mmCIF syntax\.

### Parser State Machine

  **Sources:** [PdbxParser\.py L68-L73](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L68-L73) [PdbxParser\.py L114-L335](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L114-L335)

### Tokenization Process

 The tokenizer uses a comprehensive regex pattern to identify different syntax elements:

  The tokenizer handles several syntax elements:

 - Category\-attribute pairs \(`_category.attribute`\)
- Single and double quoted strings
- Multi\-line semicolon\-delimited strings
- Comments \(which are discarded\)
- Unquoted words

 **Sources:** [PdbxParser\.py L337-L410](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L337-L410) [PdbxParser\.py L351-L363](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L351-L363)

## Data Flow and Processing

 The complete processing pipeline from file input to structured data follows this flow:

  **Sources:** [PdbxParser\.py L74-L86](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L74-L86) [PdbxParser\.py L337-L410](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L337-L410) [PdbxParser\.py L114-L335](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L114-L335)

## Writer Functionality

 The `PdbxWriter` class provides output formatting capabilities with support for both item\-value pairs and tabular formats\.

### Output Formatting

 The writer supports two primary output formats:

 1. **Item\-Value Format** \- For single\-row categories:   ``` _category.attribute1    value1 _category.attribute2    value2 ```
2. **Loop/Table Format** \- For multi\-row categories:   ``` loop_ _category.attribute1 _category.attribute2 value1  value2 value3  value4 ```

 The writer automatically determines the appropriate format based on the number of rows in each category and applies proper quoting rules for string values\.

 **Sources:** [PdbxParser\.py L489-L639](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L489-L639) [PdbxParser\.py L554-L575](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L554-L575) [PdbxParser\.py L577-L638](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxParser.py#L577-L638)

## Data Type Handling

 The system includes sophisticated data type detection and formatting capabilities:

| Data Type | Format Code | Description |
| --- | --- | --- |
| DT\_NULL\_VALUE | FT\_NULL\_VALUE | Null values \(\., ?\) |
| DT\_INTEGER | FT\_NUMBER | Integer values |
| DT\_FLOAT | FT\_NUMBER | Floating point values |
| DT\_UNQUOTED\_STRING | FT\_UNQUOTED\_STRING | Simple strings |
| DT\_SINGLE\_QUOTED\_STRING | FT\_QUOTED\_STRING | Single quoted strings |
| DT\_DOUBLE\_QUOTED\_STRING | FT\_QUOTED\_STRING | Double quoted strings |
| DT\_MULTI\_LINE\_STRING | FT\_MULTI\_LINE\_STRING | Semicolon\-delimited strings |

 The system automatically selects appropriate quoting based on content analysis, preferring double quotes when possible to avoid embedded quote conflicts\.

 **Sources:** [PdbxContainers\.py L605-L657](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L605-L657) [PdbxContainers\.py L658-L703](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L658-L703) [PdbxContainers\.py L307-L311](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L307-L311)

## Usage Patterns

### Basic Reading Pattern

### Category Data Access

 The `DataCategory` class provides multiple access patterns:

  **Sources:** [PdbxContainers\.py L314-L331](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L314-L331) [PdbxContainers\.py L452-L503](https://github.com/uw-ipd/RoseTTAFold2/blob/4b273b95/input_prep/pdbx/reader/PdbxContainers.py#L452-L503)

---
*Source: [https://deepwiki.com/uw-ipd/RoseTTAFold2/7.1-pdbxmmcif-processing](https://deepwiki.com/uw-ipd/RoseTTAFold2/7.1-pdbxmmcif-processing) on DeepWiki*