---
title: "Template Processing"
source: deepwiki.com
owner: jcheongs
repo: alphafold-multimer
url: https://deepwiki.com/jcheongs/alphafold-multimer/4.3-template-processing
---
# Template Processing

# Template Processing

> **Relevant source files**
> - [alphafold/data/parsers\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py)
> - [alphafold/data/templates\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py)

 This page documents how structural template hits are validated and converted into numerical feature arrays for the AlphaFold model\. It covers the classes and functions in [alphafold/data/templates\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py) the `TemplateHit` data structure from [alphafold/data/parsers\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py) and the complete chain from raw search hits to the final `TEMPLATE_FEATURES` arrays\.

 For the upstream tools that produce template hits \(HHSearch, Hmmsearch\), see [MSA Generation Tools](https://deepwiki.com/jcheongs/alphafold-multimer/4.2-msa-generation-tools)\. For the data pipeline context in which template features are assembled alongside MSA features, see [Monomer Data Pipeline](https://deepwiki.com/jcheongs/alphafold-multimer/4.1-monomer-data-pipeline) and [Multimer Data Pipeline](https://deepwiki.com/jcheongs/alphafold-multimer/4.4-multimer-data-pipeline)\. For parsing of the raw hit file formats \(`.hhr`, hmmsearch A3M\), see [Data Formats and Parsers](https://deepwiki.com/jcheongs/alphafold-multimer/4.5-data-formats-and-parsers)\.

---

## Overview

 Template processing converts a list of `TemplateHit` objects — produced by HHSearch or Hmmsearch — into a fixed\-shape dictionary of numerical arrays that the model uses as structural priors\. The process involves:

 1. **Prefiltering** each hit \(date, alignment ratio, duplicate detection\)
2. **Index mapping** from query sequence positions to template hit positions
3. **mmCIF parsing** to load the actual PDB structure
4. **Atom position extraction** into atom37 representation
5. **Alignment\-based placement** of template atoms at query\-indexed positions
6. **Stacking** all passing hits into batch arrays

 **Template diagram: End\-to\-end flow**

  Sources: [templates\.py L686-L791](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L686-L791) [templates\.py L800-L1010](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L800-L1010)

---

## Key Data Structures

### `TemplateHit`

 Defined in [parsers\.py L55-L66](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py#L55-L66) `TemplateHit` is a frozen dataclass produced by `parse_hhr` \(HHSearch\) or `parse_hmmsearch_a3m` \(Hmmsearch\)\.

| Field | Type | Description |
| --- | --- | --- |
| index | int | Hit rank within the search result |
| name | str | Template identifier, e\.g\. "4abc\_A" \(PDB ID \+ chain\) |
| aligned\_cols | int | Number of aligned columns in the hit |
| sum\_probs | Optional\[float\] | Sum of posterior probabilities \(used for sorting; None in Hmmsearch hits\) |
| query | str | The aligned query subsequence \(may contain gaps \-\) |
| hit\_sequence | str | The aligned hit subsequence \(may contain gaps \-\) |
| indices\_query | List\[int\] | Per\-position index into the original query sequence; \-1 for gaps |
| indices\_hit | List\[int\] | Per\-position index into the hit sequence; \-1 for gaps |

### `TEMPLATE_FEATURES` Schema

 Defined at [templates\.py L88-L95](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L88-L95) this dictionary specifies the six feature arrays that every successful template contributes to:

| Feature name | dtype | Shape \(per hit\) |
| --- | --- | --- |
| template\_aatype | float32 | \[num\_res, 22\] — one\-hot using HHBLITS\_AA\_TO\_ID encoding |
| template\_all\_atom\_masks | float32 | \[num\_res, 37\] — atom37 mask |
| template\_all\_atom\_positions | float32 | \[num\_res, 37, 3\] — atom37 coordinates \(Å\) |
| template\_domain\_names | object | \[1\] — bytes string, e\.g\. b"4abc\_a" |
| template\_sequence | object | \[1\] — bytes string of aligned residues, \- for gaps |
| template\_sum\_probs | float32 | \[1\] — copied from hit\.sum\_probs, or 0 if absent |

 After processing all hits, each feature is stacked on axis 0 to yield shape `[num_templates, ...]`\.

### `SingleHitResult` and `TemplateSearchResult`

 [templates\.py L672-L677](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L672-L677) and [templates\.py L793-L797](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L793-L797):

 - `SingleHitResult` — a frozen dataclass with fields `features`, `error`, `warning`\. `features` is `None` if the hit was rejected or failed\.
- `TemplateSearchResult` — the return type of `get_templates()`, containing `features` \(the stacked dict\), `errors` \(list of strings\), and `warnings` \(list of strings\)\.

---

## `TemplateHitFeaturizer` \(Abstract Base Class\)

 Defined at [templates\.py L800-L867](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L800-L867) All concrete featurizers share a common constructor:

| Parameter | Type | Purpose |
| --- | --- | --- |
| mmcif\_dir | str | Directory of \.cif files indexed by PDB ID |
| max\_template\_date | str | ISO 8601 date string \(e\.g\. "2021\-11\-01"\); templates after this date are excluded |
| max\_hits | int | Maximum number of accepted templates to return |
| kalign\_binary\_path | str | Path to kalign executable for sequence realignment |
| release\_dates\_path | Optional\[str\] | Path to precomputed PDB release date file \(avoids repeated mmCIF parsing\) |
| obsolete\_pdbs\_path | Optional\[str\] | Path to PDB obsolete entries file |
| strict\_error\_check | bool | If True, date/duplicate errors are promoted to hard errors |

 The constructor validates that `mmcif_dir` contains `.cif` files, parses `max_template_date` into a `datetime` object, and optionally loads `_release_dates` and `_obsolete_pdbs` maps via `_parse_release_dates` and `_parse_obsolete`\.

 The single abstract method is:

```
get_templates(query_sequence: str, hits: Sequence[TemplateHit]) -> TemplateSearchResult
```

 Sources: [templates\.py L800-L868](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L800-L868)

---

## Concrete Implementations

 **Diagram: Class hierarchy and search\-tool alignment**

  Sources: [templates\.py L870-L1010](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L870-L1010)

### `HhsearchHitFeaturizer`

 Used with monomer presets\. Hits are sorted descending by `sum_probs` before processing\. Processing stops once `num_hits >= max_hits` successful hits have been collected\. Defined at [templates\.py L870-L929](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L870-L929)

### `HmmsearchHitFeaturizer`

 Used with the multimer preset\. Key behavioral differences from `HhsearchHitFeaturizer`:

 - Sorting by `sum_probs` is skipped when hits have no `sum_probs` value \(Hmmsearch A3M hits always have `sum_probs=None`\)
- Deduplication is performed by `template_sequence` content: a hit whose aligned sequence was already seen is skipped
- When no hits pass, a **zero\-filled default template** is returned rather than an empty array — this ensures a valid shape for model consumption

 The default template shape when no hits pass is `(1, num_res, ...)` with zeros and empty byte strings\. Defined at [templates\.py L932-L1010](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L932-L1010)

| Behavior | HhsearchHitFeaturizer | HmmsearchHitFeaturizer |
| --- | --- | --- |
| Sort order | sum\_probs descending | sum\_probs descending if available, else original order |
| Deduplication | None | By template\_sequence bytes |
| Empty result | Empty np\.array\(\[\], dtype=\.\.\.\) | Zero\-filled \(1, num\_res, \.\.\.\) arrays |
| Search tool source | HHSearch → \.hhr → parse\_hhr | Hmmsearch → A3M → parse\_hmmsearch\_a3m |

---

## `_process_single_hit` Internals

 [templates\.py L686-L790](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L686-L790) orchestrates the full processing pipeline for one `TemplateHit`\. Steps in order:

 **Diagram: `_process_single_hit` decision flow**

  Note: `_read_file` at [templates\.py L679-L683](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L679-L683) is decorated with `@functools.lru_cache(16)`, so repeated reads of the same `.cif` file within a run are served from memory\.

 Sources: [templates\.py L686-L790](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L686-L790)

---

## Hit Prefiltering: `_assess_hhsearch_hit`

 [templates\.py L173-L230](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L173-L230) This function performs four checks before any mmCIF file is touched:

| Check | Exception raised | Condition |
| --- | --- | --- |
| Date cutoff | DateError | Template release date \> max\_template\_date |
| Alignment ratio | AlignRatioError | aligned\_cols / len\(query\_sequence\) <= 0\.1 |
| Duplicate detection | DuplicateError | Template sequence is substring of query and covers \> 95% of query length |
| Minimum length | LengthError | Template sequence \(gap\-stripped\) has fewer than 10 residues |

 All four are subclasses of `PrefilterError` [templates\.py L68-L85](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L68-L85) In non\-strict mode, all prefilter rejections result in `SingleHitResult(features=None, error=None, warning=None)` — the hit is silently skipped\. In strict mode \(`strict_error_check=True`\), `DateError` and `DuplicateError` are promoted to the `error` field\.

 The date check relies on `_is_after_cutoff` [templates\.py L108-L129](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L108-L129) which uses a preloaded `release_dates` dict where available, and skips the check \(returning `False`\) when the PDB ID is not found — to avoid false positives in the prefilter\.

---

## Query\-to\-Hit Index Mapping: `_build_query_to_hit_index_mapping`

 [templates\.py L615-L669](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L615-L669) This function converts the gapped alignment columns from a `TemplateHit` into a dictionary `{query_position: hit_position}` using 0\-based indices into the **ungapped** sequences\.

 The alignment in a `TemplateHit` covers only the matched region of the query; `hit.query` is a substring of the full query\. The function corrects for this by:

 1. Stripping gaps from `hit.query` to get `hhsearch_query_sequence`
2. Finding the offset of `hhsearch_query_sequence` within `original_query_sequence` via `str.find()`
3. Re\-zeroing `indices_hit` and `indices_query` by subtracting their respective minimums
4. Adding the offset to all query indices so they reference positions in the full `original_query_sequence`

 Pairs where either index is `-1` \(gap character\) are skipped\. The resulting mapping is used directly in `_extract_template_features` to place atoms at the correct query positions\.

---

## Feature Extraction: `_extract_template_features`

 [templates\.py L485-L612](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L485-L612) Given a parsed `MmcifObject` and the query\-to\-hit index mapping, this function produces one template's contribution to `TEMPLATE_FEATURES`\.

### Chain Lookup: `_find_template_in_pdb`

 [templates\.py L233-L294](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L233-L294) Locates the correct chain in the mmCIF by trying three strategies in order:

 1. Exact match on both chain ID and sequence \(substring\)
2. Exact match on sequence only \(across all chains\)
3. Fuzzy match where `X` in the template sequence acts as a wildcard

 Returns `(chain_sequence, chain_id, mapping_offset)` where `mapping_offset` is the start position of the template subsequence within the chain\.

 If none of these succeed, `SequenceNotInTemplateError` is raised, which triggers a fallback to `_realign_pdb_template_to_query`\.

### Realignment Fallback: `_realign_pdb_template_to_query`

 [templates\.py L297-L406](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L297-L406) Called when the sequence in PDB70 \(used by HHSearch\) differs from the sequence in the actual mmCIF file \(PDB is updated more frequently than PDB70\)\. Steps:

 1. Runs `Kalign.align([old_template_sequence, new_template_sequence])`
2. Parses the aligned output via `parsers.parse_a3m`
3. Builds `old_to_new_template_mapping` from the pairwise alignment
4. Rejects the hit if fewer than 90% of residues match \(min\-length normalized\)
5. Remaps the original `query_to_template_mapping` through the new alignment

### Atom Position Extraction: `_get_atom_positions`

 [templates\.py L430-L482](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L430-L482) Iterates over residues in the chain's SEQRES sequence and populates atom37 arrays:

 - `all_positions`: shape `[num_res, 37, 3]`, float32
- `all_positions_mask`: shape `[num_res, 37]`, int64

 Special cases handled:

 - Missing residues \(`res_at_position.is_missing`\) are left as zeros
- Selenomethionine \(`MSE`\) selenium \(`SE`\) is placed at the sulphur \(`SD`\) index
- Arginine NH1/NH2 naming errors are corrected based on distance to CD

 Residue distance validation via `_check_residue_distances` [templates\.py L409-L427](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L409-L427) rejects templates where adjacent Cα atoms are more than 150 Å apart \(`CaDistanceError`\)\.

### Alignment\-Based Atom Placement

 [templates\.py L575-L612](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L575-L612) The core loop of `_extract_template_features`:

 1. Initializes output arrays of length `len(query_sequence)` with zeros and gap characters \(`-`\)
2. For each `(query_idx, hit_idx)` pair in the mapping, copies: - `all_atom_positions[hit_idx + mapping_offset]` → `templates_all_atom_positions[query_idx]` - `all_atom_masks[hit_idx + mapping_offset]` → `templates_all_atom_masks[query_idx]` - `template_sequence[hit_idx]` → `output_templates_sequence[query_idx]`
3. Converts the output sequence to one\-hot via `residue_constants.sequence_to_onehot(..., HHBLITS_AA_TO_ID)`
4. Rejects the template if total atom mask sum < 5 \(`TemplateAtomMaskAllZerosError`\)

 **Diagram: Index mapping through `_extract_template_features`**

  Sources: [templates\.py L485-L612](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L485-L612)

---

## Error Hierarchy

| Exception class | Where raised | Effect in \_process\_single\_hit |
| --- | --- | --- |
| PrefilterError subclasses | \_assess\_hhsearch\_hit | Hit skipped \(warning in strict mode for some\) |
| NoChainsError | \_extract\_template\_features | Converted to warning |
| NoAtomDataInTemplateError | \_extract\_template\_features | Converted to warning |
| TemplateAtomMaskAllZerosError | \_extract\_template\_features | Converted to warning |
| SequenceNotInTemplateError | \_find\_template\_in\_pdb | Triggers \_realign\_pdb\_template\_to\_query |
| QueryToTemplateAlignError | \_realign\_pdb\_template\_to\_query | Hard error |
| CaDistanceError | \_check\_residue\_distances | Wrapped into NoAtomDataInTemplateError |
| MultipleChainsError | \_get\_atom\_positions | Wrapped into NoAtomDataInTemplateError |

 Sources: [templates\.py L35-L95](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L35-L95)

---

## Obsolete PDB Handling

 [templates\.py L132-L152](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L132-L152) `_parse_obsolete` reads the PDB `obsolete.dat` file\. The resulting dict maps:

 - `old_pdb_id → new_pdb_id` \(replaced entries\)
- `old_pdb_id → None` \(removed entries\)

 In `_process_single_hit`, if a hit's PDB code maps to `None` it is skipped immediately with a warning\. If it maps to a replacement ID, that replacement is used for the release date lookup and passed to `_assess_hhsearch_hit`\. The mmCIF file is still looked up using the original hit code from the filename\.

---

## Release Date Handling

 [templates\.py L155-L170](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L155-L170) `_parse_release_dates` reads a plain\-text file with lines of the form `PDBID: YYYY-MM-DD`\. The resulting dict is checked by `_is_after_cutoff` before any mmCIF parsing occurs \(fast prefilter\)\. After mmCIF parsing, the release date from the mmCIF header is also checked directly in `_process_single_hit` [templates\.py L743-L752](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L743-L752) as a secondary guard\.

 Sources: [templates\.py L108-L170](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/templates.py#L108-L170)

---
*Source: [https://deepwiki.com/jcheongs/alphafold-multimer/4.3-template-processing](https://deepwiki.com/jcheongs/alphafold-multimer/4.3-template-processing) on DeepWiki*