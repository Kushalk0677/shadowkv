# External Artifact Manifest

This repository should contain clean source, documentation, curated result tables, and small diagnostic examples. Large transfer packages and raw working archives should stay outside git and be attached to a GitHub Release or stored in a lab artifact store.

## Keep Out Of Git

| Artifact | Status | Reason |
|---|---|---|
| `p100_transfer/` | external transfer package | Contains a self-contained runner copy, logs, tarballs, and historical scripts. Public repo now has `experiments/run_p100_isolated_sweep.py` and `docs/reproducing_p100.md`. |
| `rtx6000_run_package.zip` | historical handoff archive | Contains pycache, egg-info, backups, session files, and bundled source. Do not commit. |
| `blackwell_semantic_n128_pipeline.zip` | retired overlay package | Too small and incomplete for source control. Superseded by the public Blackwell runner and the full sendable package. |
| `shadowkv_blackwell_semantic_n128_fullrepo_*.zip` | sendable full repo package | Useful for handing to a machine operator, but should be attached as a release artifact rather than committed. |
| `v10/` | private working archive | Contains scratch experiments, old packages, profiles, notebooks, and intermediate outputs. Extract only curated docs/scripts/results. |
| `.stratum/artifact_extraction/` | raw artifact extraction archive | Contains reviewer-gap runs, fidelity outputs, and runtime replicate bundles. Curate summaries into `results/` or `runtime_experiments/` before publishing. |

## Public Repo Equivalents

| Need | Public file |
|---|---|
| Blackwell semantic n=128 run | `experiments/run_blackwell_semantic_n128.py` and `docs/reproducing_blackwell.md` |
| P100 rerun | `experiments/run_p100_isolated_sweep.py` and `docs/reproducing_p100.md` |
| Controlled T4/P100 results | `results/controlled_results/` |
| Process-isolated sanity checks | `results/realistic_results/` |
| Fidelity examples | `results/fidelity_examples/` |
| Runtime SGLang/vLLM/LMCache summaries | `runtime_experiments/` |

## Rule Of Thumb

If an artifact contains generated logs, cache directories, bundled source copies, or old session notes, do not commit it. If it contains a small script or a result table that someone can understand and rerun from the public repo, extract that piece and document where it came from.
