- **Input:** single-chain amino-acid sequence (`aatype`) + MSA-derived features (`msa_feat`, `extra_msa`), optionally template features and prior-cycle recycled representations (`prev_pos`, `prev_pair`, `prev_msa_first_row`).
- **Output:** a single 3D atomic structure (`final_atom_positions`) accompanied by per-residue confidence (`plddt`) and a pairwise distance-distribution (`distogram`), not a binary contact map. In pTM/multimer models, additionally `predicted_aligned_error` and `ptm`/`iptm`; chain-level (intra vs. inter) distinctions in multimer mode are handled via `asym_id`/`entity_id` features and separate `intra_chain_fape`/`interface_fape` loss terms, not via a distinct "contact map" output.

## Reimplementations / Resources

- OpenFold: https://github.com/aqlaboratory/openfold
- lucidrains/alphafold2: https://github.com/lucidrains/alphafold2
- ChrisHayduk/minAlphaFold2: https://github.com/ChrisHayduk/minAlphaFold2
- ocx-lab/OpenComplex: https://github.com/ocx-lab/OpenComplex
- FastFold: https://github.com/hpcaitech/FastFold