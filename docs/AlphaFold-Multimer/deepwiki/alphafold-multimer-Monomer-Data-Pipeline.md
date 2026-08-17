---
title: "Monomer Data Pipeline"
source: deepwiki.com
owner: jcheongs
repo: alphafold-multimer
url: https://deepwiki.com/jcheongs/alphafold-multimer/4.1-monomer-data-pipeline
---
# Monomer Data Pipeline

# Monomer Data Pipeline

> **Relevant source files**
> - [alphafold/data/parsers\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py)
> - [alphafold/data/pipeline\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py)

## Purpose and Scope

 This page documents the monomer data pipeline implemented in [alphafold/data/pipeline\.py](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py) It covers the `DataPipeline` class, all MSA search steps, the template search branch, and the feature assembly functions that produce the final `FeatureDict` passed to the model\.

 This pipeline is used directly for monomer predictions \(`monomer`, `monomer_ptm`, `monomer_casp14` presets\) and also as a per\-chain subroutine inside the multimer pipeline\. For details on how the multimer pipeline wraps and extends this output, see [4\.4](https://deepwiki.com/jcheongs/alphafold-multimer/4.4-multimer-data-pipeline)\. For the bioinformatics tool wrappers themselves \(Jackhmmer, HHBlits, HHSearch, Hmmsearch\), see [4\.2](https://deepwiki.com/jcheongs/alphafold-multimer/4.2-msa-generation-tools)\. For template featurization, see [4\.3](https://deepwiki.com/jcheongs/alphafold-multimer/4.3-template-processing)\.

---

## Key Types

| Symbol | Location | Description |
| --- | --- | --- |
| DataPipeline | alphafold/data/pipeline\.py116\-248 | Main class; owns all tool runners and drives the pipeline |
| FeatureDict | alphafold/data/pipeline\.py32 | Type alias for MutableMapping\[str, np\.ndarray\] |
| TemplateSearcher | alphafold/data/pipeline\.py33 | Union of HHSearch and Hmmsearch |
| make\_sequence\_features | alphafold/data/pipeline\.py36\-50 | Builds sequence\-level features from the raw FASTA |
| make\_msa\_features | alphafold/data/pipeline\.py53\-89 | Merges and encodes MSA hits from all sources |
| run\_msa\_tool | alphafold/data/pipeline\.py92\-113 | Caching wrapper around any MSA runner |

---

## DataPipeline Constructor

 `DataPipeline.__init__` [pipeline\.py L119-L153](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L119-L153) accepts the following parameters:

| Parameter | Type | Purpose |
| --- | --- | --- |
| jackhmmer\_binary\_path | str | Path to the Jackhmmer binary |
| hhblits\_binary\_path | str | Path to the HHBlits binary |
| uniref90\_database\_path | str | UniRef90 FASTA database |
| mgnify\_database\_path | str | MGnify FASTA database |
| bfd\_database\_path | Optional\[str\] | BFD database \(required when use\_small\_bfd=False\) |
| uniclust30\_database\_path | Optional\[str\] | UniClust30 database \(required when use\_small\_bfd=False\) |
| small\_bfd\_database\_path | Optional\[str\] | Reduced BFD database \(required when use\_small\_bfd=True\) |
| template\_searcher | TemplateSearcher | Either HHSearch or Hmmsearch instance |
| template\_featurizer | TemplateHitFeaturizer | Converts template hits to numeric arrays |
| use\_small\_bfd | bool | Selects between reduced and full database paths |
| mgnify\_max\_hits | int | Cap on MGnify sequences \(default: 501\) |
| uniref\_max\_hits | int | Cap on UniRef90 sequences \(default: 10000\) |
| use\_precomputed\_msas | bool | If True, reuse previously saved MSA files |

 The constructor conditionally instantiates either `jackhmmer_small_bfd_runner` or `hhblits_bfd_uniclust_runner` depending on `use_small_bfd`:

 **DataPipeline: BFD Runner Selection**

  Sources: [pipeline\.py L119-L153](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L119-L153)

---

## process\(\) Method Flow

 `DataPipeline.process(input_fasta_path, msa_output_dir)` [pipeline\.py L155-L248](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L155-L248) accepts a single\-sequence FASTA file and an output directory\. It raises `ValueError` if the FASTA contains more than one sequence\.

 The method proceeds in four logical phases:

 1. **MSA generation** — run Jackhmmer against UniRef90 and MGnify; run either Jackhmmer against small BFD or HHBlits against BFD\+UniClust30
2. **Template search** — derive a cleaned MSA from UniRef90 hits and pass it to the `template_searcher`
3. **Feature assembly** — encode the sequence, merge MSAs, featurize templates
4. **Return** — merge all sub\-dictionaries into a single `FeatureDict`

 **process\(\) Method: Step\-by\-Step Data Flow**

  Sources: [pipeline\.py L155-L248](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L155-L248)

---

## MSA Sequence Caps

 Jackhmmer's streaming mode accepts a `max_sequences` argument to limit how many sequences it retains from the database\. The caps configured in the constructor are applied inside `run_msa_tool`:

| Database | Runner | Default Cap | Output Format |
| --- | --- | --- | --- |
| UniRef90 | jackhmmer\_uniref90\_runner | uniref\_max\_hits=10000 | Stockholm \(\.sto\) |
| MGnify | jackhmmer\_mgnify\_runner | mgnify\_max\_hits=501 | Stockholm \(\.sto\) |
| Small BFD | jackhmmer\_small\_bfd\_runner | No cap | Stockholm \(\.sto\) |
| BFD \+ UniClust30 | hhblits\_bfd\_uniclust\_runner | No cap | A3M \(\.a3m\) |

 Sources: [pipeline\.py L92-L113](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L92-L113) [pipeline\.py L167-L226](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L167-L226)

---

## run\_msa\_tool: Caching Helper

 `run_msa_tool` [pipeline\.py L92-L113](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L92-L113) wraps any MSA runner to provide transparent file\-based caching:

 - If `use_precomputed_msas=True` **and** `msa_out_path` exists on disk, the tool is **not** re\-run\. The cached file is read back instead\.
- When reading a cached `.sto` file with a `max_sto_sequences` cap, `parsers.truncate_stockholm_msa()` is applied in\-memory to avoid loading the full file\.
- Otherwise, the runner's `query()` method is called\. For Stockholm format with a cap, it uses the two\-argument `query(input_fasta_path, max_sto_sequences)` form \(Jackhmmer's streaming interface\)\. The result is written to disk\.

```python
run_msa_tool(msa_runner, input_fasta_path, msa_out_path, msa_format, use_precomputed_msas, max_sto_sequences)
   │
   ├─ use_precomputed_msas=True AND file exists → read from disk
   │     └─ sto + max_sto_sequences? → truncate_stockholm_msa()
   │
   └─ otherwise → msa_runner.query(...)  → write to disk
         └─ sto + max_sto_sequences? → query(path, max_sto_sequences)
```

 Sources: [pipeline\.py L92-L113](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L92-L113)

---

## Template Search Using UniRef90 MSA

 The pipeline reuses the UniRef90 MSA \(not the raw query\) as the input to the template searcher\. Before querying, the Stockholm string is cleaned to remove redundancy and uninformative columns:

 1. `parsers.deduplicate_stockholm_msa(msa_for_templates)` — removes sequences that are identical to another hit after masking out insertion columns relative to the query [parsers\.py L340-L372](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py#L340-L372)
2. `parsers.remove_empty_columns_from_stockholm_msa(msa_for_templates)` — strips columns where every row contains only gaps [parsers\.py L300-L337](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/parsers.py#L300-L337)

 The cleaned MSA is then dispatched based on `template_searcher.input_format`:

 - `'sto'` → passed directly \(used by `HHSearch`\)
- `'a3m'` → first converted via `parsers.convert_stockholm_to_a3m()`, then passed \(used by `Hmmsearch`\)

 Template hits are saved to `pdb_hits.hhr` or `pdb_hits.a3m` depending on the searcher's `output_format`\.

 Sources: [pipeline\.py L184-L207](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L184-L207)

---

## make\_sequence\_features

 [pipeline\.py L36-L50](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L36-L50)

 Takes the raw `sequence` string, `description` line, and `num_res` count\. Returns:

| Feature Key | Shape | dtype | Description |
| --- | --- | --- | --- |
| aatype | \(num\_res, 21\) | int32 | One\-hot encoding using restype\_order\_with\_x; unknowns mapped to X |
| between\_segment\_residues | \(num\_res,\) | int32 | All zeros \(single\-chain\) |
| domain\_name | \(1,\) | object | UTF\-8 encoded description string |
| residue\_index | \(num\_res,\) | int32 | \[0, 1, 2, \.\.\., num\_res\-1\] |
| seq\_length | \(num\_res,\) | int32 | num\_res repeated num\_res times |
| sequence | \(1,\) | object | UTF\-8 encoded sequence string |

 Sources: [pipeline\.py L36-L50](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L36-L50)

---

## make\_msa\_features

 [pipeline\.py L53-L89](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L53-L89)

 Accepts a sequence of `parsers.Msa` objects \(one per source database\) and merges them into a single feature set\. Sequences are deduplicated globally across all sources — a sequence seen in UniRef90 will be skipped if it appears again in BFD or MGnify\.

 The MSA order passed in `process()` is:

  For each unique sequence, species and UniProt accession identifiers are extracted from the sequence description via `msa_identifiers.get_identifiers()`\.

 Output features:

| Feature Key | Shape | dtype | Description |
| --- | --- | --- | --- |
| msa | \(num\_alignments, num\_res\) | int32 | Integer\-encoded residues using HHBLITS\_AA\_TO\_ID |
| deletion\_matrix\_int | \(num\_alignments, num\_res\) | int32 | Deletion counts per position |
| num\_alignments | \(num\_res,\) | int32 | Total deduplicated alignment count \(repeated\) |
| msa\_uniprot\_accession\_identifiers | \(num\_alignments,\) | object | Per\-sequence UniProt accession |
| msa\_species\_identifiers | \(num\_alignments,\) | object | Per\-sequence species identifier |

 Sources: [pipeline\.py L53-L89](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L53-L89)

---

## Final FeatureDict Composition

 `process()` returns a single merged dictionary:

 **FeatureDict Output Assembly**

  This combined dictionary is returned to `run_alphafold.predict_structure()` for the monomer path, or to `pipeline_multimer.DataPipeline._process_single_chain()` for the multimer path\.

 Sources: [pipeline\.py L232-L248](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L232-L248)

---

## File Outputs

 For each call to `process()`, the following files are written to `msa_output_dir`:

| Filename | Producer | Format |
| --- | --- | --- |
| uniref90\_hits\.sto | jackhmmer\_uniref90\_runner | Stockholm |
| mgnify\_hits\.sto | jackhmmer\_mgnify\_runner | Stockholm |
| small\_bfd\_hits\.sto | jackhmmer\_small\_bfd\_runner | Stockholm \(reduced only\) |
| bfd\_uniclust\_hits\.a3m | hhblits\_bfd\_uniclust\_runner | A3M \(full only\) |
| pdb\_hits\.hhr or pdb\_hits\.a3m | template\_searcher | HHR or A3M |

 These files serve as a cache when `use_precomputed_msas=True` on subsequent runs\.

 Sources: [pipeline\.py L167-L226](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L167-L226) [pipeline\.py L198-L201](https://github.com/jcheongs/alphafold-multimer/blob/8e419501/alphafold/data/pipeline.py#L198-L201)

---
*Source: [https://deepwiki.com/jcheongs/alphafold-multimer/4.1-monomer-data-pipeline](https://deepwiki.com/jcheongs/alphafold-multimer/4.1-monomer-data-pipeline) on DeepWiki*