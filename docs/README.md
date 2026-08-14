# docs/ — per-tool long-content store

The README stays lean. Each tool's detailed written content lives in **one folder per tool** under `docs/`, named `docs/<tool>/`. All per-tool derived content (abstract + method, plus deepwiki/zread code annotations) lives in that folder.

## Layout

```
docs/
  README.md            # this convention doc
  <tool>/
    README.md          # abstract + method ("Read more →" target)
    deepwiki/          # DeepWiki agent's exported annotation folder (many md files)
      *.md
    zread/             # ZRead agent's exported code-reading notes folder (many md files)
      *.md
```

- The `docs/<tool>/README.md` uses two heading anchors, linked from the README:
  - `<ToolName> Abstract` (abstract)
  - `<ToolName> Model/Method` (model / method)
- `deepwiki/` and `zread/` are **folders** because the web agents (DeepWiki / ZRead) export whole directory trees of markdown (one md per module/file). Create them only when you actually annotate that tool.
- For unpublished tools, add a Note (⚠️ not peer-reviewed; experimental) and cite a verifiable source.
- The README entry links to `docs/<tool>/` via a `Docs:` line; folders not yet written up carry `*to be written*`.
- **Reimplementations, extra resources and other secondary links** referenced by a tool are listed in its `docs/<tool>/README.md` (a `Reimplementations / Resources` section) — never in the main README entry, which keeps only the official Code + Web server.

## Template (`docs/<tool>/README.md`)

```markdown
# <ToolName>

## <ToolName> Abstract

<full abstract from paper, 1-3 paragraphs>

## <ToolName> Model/Method

<model architecture, input/output, training data, post-processing>

## Reimplementations / Resources  (optional)

- OpenFold: <link>
- lucidrains/alphafold2: <link>
```
