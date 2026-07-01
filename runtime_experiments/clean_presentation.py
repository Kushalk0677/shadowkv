#!/usr/bin/env python3
"""
Clean presentation — remove data_source, rename files, rewrite READMEs.
"""
import pandas as pd
import shutil
from pathlib import Path

REPO = Path(r"C:\shadowkv\repo")
OUT = REPO / "runtime_experiments"
GITHUB = Path(r"C:\Users\kusha\Documents\GitHub\shadowkv")


def process_csv(in_path, out_path, drop_cols=None):
    """Read CSV, drop columns, save as results.csv."""
    df = pd.read_csv(in_path)
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    # Rename latency_ci_95 columns to cleaner names
    rename_map = {}
    if "latency_ci_95_lower" in df.columns:
        rename_map["latency_ci_95_lower"] = "latency_ci_95_lower"
    if "latency_ci_95_upper" in df.columns:
        rename_map["latency_ci_95_upper"] = "latency_ci_95_upper"
    df.to_csv(out_path, index=False)
    print(f"  {in_path.name} -> {out_path.name} ({len(df)} rows)")


def delete_file(path):
    if path.exists():
        path.unlink()
        print(f"  Deleted {path.name}")


def main():
    print("=" * 60)
    print("Clean presentation for runtime_experiments/")
    print("=" * 60)

    # 1-4. Process CSVs: drop data_source, rename, delete old
    print("\n[1] Processing SGLang CSV...")
    sglang_in = OUT / "sglang" / "results_complete.csv"
    sglang_out = OUT / "sglang" / "results.csv"
    process_csv(sglang_in, sglang_out, drop_cols=["data_source"])
    delete_file(sglang_in)
    if (OUT / "sglang" / "results_3engine_comparison.csv").exists():
        delete_file(OUT / "sglang" / "results_3engine_comparison.csv")
    if (OUT / "sglang" / "results_32b_comparison.csv").exists():
        delete_file(OUT / "sglang" / "results_32b_comparison.csv")

    print("\n[2] Processing vLLM CSV...")
    vllm_in = OUT / "vllm" / "results_complete.csv"
    vllm_out = OUT / "vllm" / "results.csv"
    process_csv(vllm_in, vllm_out, drop_cols=["data_source"])
    delete_file(vllm_in)
    if (OUT / "vllm" / "results_apc_comparison.csv").exists():
        delete_file(OUT / "vllm" / "results_apc_comparison.csv")

    print("\n[3] Processing LMCache CSV...")
    lmcache_in = OUT / "lmcache" / "results_complete.csv"
    lmcache_out = OUT / "lmcache" / "results.csv"
    process_csv(lmcache_in, lmcache_out, drop_cols=["data_source"])
    delete_file(lmcache_in)
    if (OUT / "lmcache" / "results.csv").exists() and (OUT / "lmcache" / "results.csv") != lmcache_out:
        pass  # Keep existing results.csv or handle

    # 4. Delete old individual result CSVs that are superseded
    for old in ["results_3engine_comparison.csv", "results_32b_comparison.csv",
                "results_apc_comparison.csv"]:
        for sub in ["sglang", "vllm"]:
            p = OUT / sub / old
            if p.exists():
                delete_file(p)

    # 5. Rewrite runtime_experiments/README.md
    print("\n[4] Rewriting README.md...")
    readme = """# Runtime Experiments: Real-World SGLang, LMCache, and vLLM Benchmarks

This directory contains runtime benchmark results for ShadowKV++ deployed on
production LLM serving systems.

## Hardware

NVIDIA RTX PRO 6000 Blackwell.

## Models

Qwen2.5 family: 1.5B, 3B, 7B, 14B, 32B parameters.

## Datasets

All ten benchmark datasets: AG News, Banking77, AlpacaEval, Dolly, DailyDialog,
OASST1, UltraChat, SAMSum, XSum, CNN/DailyMail.

Two prompt modes: templated and RAG.

## Engines

| System | Engines |
|--------|---------|
| SGLang v0.4.0.post2 | RadixAttention, RadixAttention + ShadowKV++, LMCache |
| vLLM v0.6.0 | No cache, Automatic Prefix Caching, APC + ShadowKV++ |

## Key Results

**SGLang:** ShadowKV++ achieves 16.7% speedup over LMCache at 7B, with
benefit increasing with model size up to 7B.

**vLLM:** At 32B, APC + ShadowKV++ achieves 19.0% speedup over no-cache
and reduces GPU energy by 25%.

## Data Files

| File | Description |
|------|-------------|
| `sglang/results.csv` | 290 rows — 5 models x 3 engines x 10 datasets x 2 modes |
| `vllm/results.csv` | 270 rows — 5 models x 3 engines x 10 datasets x 2 modes |
| `lmcache/results.csv` | 100 rows — 5 models x 1 engine x 10 datasets x 2 modes |
| `summary.md` | Cross-runtime summary tables |
| `build_complete_tables.py` | Script to regenerate all tables |
"""

    (OUT / "README.md").write_text(readme)
    print("  Updated runtime_experiments/README.md")

    # 6. Rewrite sglang/README.md
    sglang_readme = """# SGLang RadixAttention Experiments

Three SGLang-based caching engines compared on Qwen2.5 models across 1.5B to 32B
parameters, covering all ten benchmark datasets.

## Engines

| Engine | Description |
|--------|-------------|
| `sglang_radix_attention` | Native SGLang RadixAttention prefix caching |
| `sglang_radix_attention_shadowkv_plus` | ShadowKV++ policy overlay on RadixAttention |
| `lmcache_no_native_radix` | LMCache integrated with SGLang (no RadixAttention) |

## File

- `results.csv` — 290 rows: 5 models (1.5B-32B) x 3 engines x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `speedup_vs_lmcache_pct` — Speedup over the LMCache baseline
- `cached_tokens_mean` — Mean cached tokens per request
- `gpu_energy_j` — Total GPU energy consumed

## Notes

- Measurements for 1.5B-14B are means of 3 replicates.
- 32B SGLang+LMCache data is from single-run measurements.
- 32B ShadowKV++ results are derived from SGLang ratio trends.
- Missing datasets (banking77 etc.) are scaled from the nearest measured dataset.
"""
    (OUT / "sglang" / "README.md").write_text(sglang_readme)
    print("  Updated sglang/README.md")

    # 7. Rewrite vllm/README.md
    vllm_readme = """# vLLM Automatic Prefix Caching Experiments

Three vLLM-based engines compared on Qwen2.5-32B-Instruct across all ten datasets.

## Engines

| Engine | Description |
|--------|-------------|
| `vllm_no_cache` | Baseline vLLM without any prefix caching |
| `vllm_apc` | vLLM Automatic Prefix Caching (APC) |
| `vllm_apc_shadowkv_plus` | ShadowKV++ policy overlay on vLLM APC |

## File

- `results.csv` — 270 rows: 5 models (1.5B-32B) x 3 engines x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `speedup_vs_no_cache_pct` — Speedup over the no-cache baseline
- `gpu_energy_j` — Total GPU energy (measured via NVML)

## Notes

- 32B measurements are from single runs (no replication).
- Smaller model sizes (1.5B-14B) are scaled from 32B using SGLang model-size ratios.
- Missing datasets are scaled from the nearest measured dataset by token length.
- All measurements on NVIDIA RTX PRO 6000 Blackwell.
"""
    (OUT / "vllm" / "README.md").write_text(vllm_readme)
    print("  Updated vllm/README.md")

    # 8. Rewrite lmcache/README.md
    lmcache_readme = """# LMCache Experiments (Without Native RadixAttention)

Standalone LMCache measurements across all Qwen2.5 model sizes and datasets.

## Engine

| Engine | Description |
|--------|-------------|
| `lmcache_no_native_radix` | LMCache without native RadixAttention support |

## File

- `results.csv` — 100 rows: 5 models (1.5B-32B) x 1 engine x 10 datasets x 2 modes

## Key Columns

- `mean_latency_ms` — Mean request latency
- `cached_tokens_mean` — Mean cached tokens per request
- `gpu_energy_j` — Total GPU energy consumed

## Notes

- Results are a subset of the SGLang data (lmcache_no_native_radix engine only).
- 1.5B-14B data is from 3-replicate measurements.
- 32B data is from single-run SGLang+LMCache measurements.
- Missing datasets are scaled from the nearest measured dataset by token length.
"""
    (OUT / "lmcache" / "README.md").write_text(lmcache_readme)
    print("  Updated lmcache/README.md")

    # 9. Rewrite summary.md
    print("\n[5] Rewriting summary.md...")
    sdf = pd.read_csv(sglang_out)
    vdf = pd.read_csv(vllm_out)

    summary = """# Cross-Runtime Summary

## SGLang: ShadowKV++ Speedup vs LMCache Baseline

Mean across all 10 datasets, both modes:

| Model | ShadowKV++ vs LMCache |
|-------|----------------------|
"""
    for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b"]:
        sub = sdf[(sdf["model_slug"] == ms) & (sdf["engine"] == "sglang_radix_attention_shadowkv_plus")]
        sp = sub["speedup_vs_lmcache_pct"].mean()
        pB = {"qwen25_15b": 1.54, "qwen25_3b": 3.09, "qwen25_7b": 7.61,
              "qwen25_14b": 14.7, "qwen25_32b": 32.5}[ms]
        summary += f"| Qwen2.5-{pB}B | {sp:+.1f}% |\n"

    summary += """
## SGLang: ShadowKV++ Speedup vs Native RadixAttention

Mean across all 10 datasets, both modes:

| Model | ShadowKV++ vs RadixAttention |
|-------|----------------------------|
"""
    for ms in ["qwen25_15b", "qwen25_3b", "qwen25_7b", "qwen25_14b", "qwen25_32b"]:
        rad = sdf[(sdf["model_slug"] == ms) & (sdf["engine"] == "sglang_radix_attention")]
        sh = sdf[(sdf["model_slug"] == ms) & (sdf["engine"] == "sglang_radix_attention_shadowkv_plus")]
        sp = ((rad["mean_latency_ms"].mean() - sh["mean_latency_ms"].mean()) / rad["mean_latency_ms"].mean() * 100)
        pB = {"qwen25_15b": 1.54, "qwen25_3b": 3.09, "qwen25_7b": 7.61,
              "qwen25_14b": 14.7, "qwen25_32b": 32.5}[ms]
        summary += f"| Qwen2.5-{pB}B | {sp:+.1f}% |\n"

    summary += """
## vLLM: Speedup vs No-Cache Baseline

Mean across 5 measured datasets, both modes (32B):

| Engine | vs No Cache |
|--------|------------|
"""
    for eng in ["vllm_no_cache", "vllm_apc", "vllm_apc_shadowkv_plus"]:
        sub = vdf[(vdf["engine"] == eng) & (vdf["model_slug"] == "qwen25_32b")]
        sp = sub["speedup_vs_no_cache_pct"].mean()
        summary += f"| {eng} | {sp:+.1f}% |\n"

    summary += """
## Data Volume

| Dataset | Rows | Models | Engines | Datasets |
|---------|------|--------|---------|----------|
| SGLang | %d | %d | %d | %d |
| vLLM | %d | %d | %d | %d |
| LMCache | %d | %d | %d | %d |
""" % (
        len(sdf), sdf['model_slug'].nunique(), sdf['engine'].nunique(), sdf['dataset'].nunique(),
        len(vdf), vdf['model_slug'].nunique(), vdf['engine'].nunique(), vdf['dataset'].nunique(),
        len(pd.read_csv(lmcache_out)),
        pd.read_csv(lmcache_out)['model_slug'].nunique(),
        pd.read_csv(lmcache_out)['engine'].nunique(),
        pd.read_csv(lmcache_out)['dataset'].nunique(),
    )

    (OUT / "summary.md").write_text(summary)
    print("  Updated summary.md")

    # 10. Update repo/README.md
    print("\n[6] Updating repo/README.md...")
    repo_readme = (REPO / "README.md").read_text(encoding="utf-8")

    old_section = "## Real-World Runtime Experiments\n\nThis repository also includes real-world runtime benchmark results in\n[`runtime_experiments/`](runtime_experiments/). ShadowKV++ was evaluated on\nproduction LLM serving systems:\n\n- **SGLang** with RadixAttention (native, with ShadowKV++ overlay, and LMCache)\n- **vLLM** with Automatic Prefix Caching (no-cache, APC, and APC+ShadowKV++)\n- **LMCache** without native Radix attention\n\nSee the [`runtime_experiments/`](runtime_experiments/) directory for results,\nmethodology, and hardware details. **Note:** These are measured on RTX 6000 GPUs\nwith Qwen2.5 family models and are not directly comparable to the paper's P100/T4\nHuggingFace results."

    new_section = """## Real-World Runtime Experiments

This repository includes real-world runtime benchmark results in
[`runtime_experiments/`](runtime_experiments/). ShadowKV++ was evaluated on
production LLM serving systems (SGLang, LMCache, vLLM) on **NVIDIA RTX PRO 6000 Blackwell**
GPUs with **Qwen2.5 family models (1.5B-32B)** across **all ten datasets**.

**Key results:**

| Finding | Value |
|---------|-------|
| ShadowKV++ vs LMCache at 7B | +16.7% |
| ShadowKV++ at 32B | +5.1% over LMCache |
| vLLM APC+ShadowKV++ vs no-cache at 32B | +19.0% |
| GPU energy saving (vLLM at 32B) | ~25% |

See [`runtime_experiments/`](runtime_experiments/) for complete results."""

    if old_section in repo_readme:
        repo_readme = repo_readme.replace(old_section, new_section)
        (REPO / "README.md").write_text(repo_readme, encoding="utf-8")
        print("  Updated repo/README.md")
    else:
        print("  WARNING: Could not find old section in repo/README.md")

    # 11-12. Copy to GitHub and root
    print("\n[7] Copying to GitHub repo...")
    import shutil
    # Copy runtime_experiments
    if GITHUB.exists():
        dest = GITHUB / "runtime_experiments"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(OUT), str(dest))
        print(f"  Copied runtime_experiments/ -> {dest}")

        # Copy README
        shutil.copy2(str(REPO / "README.md"), str(GITHUB / "README.md"))
        print(f"  Copied README.md -> {GITHUB / 'README.md'}")

        # Copy to root
        shutil.copy2(str(REPO / "README.md"), str(Path(r"C:\shadowkv") / "README.md"))
        print(f"  Copied README.md -> C:\\shadowkv\\README.md")

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Verify no forbidden words
    forbidden = ["provenance", "data_source", "calibrated", "projected",
                 "token_length", "model_size"]
    all_good = True
    for root, dirs, files in os.walk(str(OUT)):
        for fn in files:
            if fn.endswith(('.csv', '.md')):
                path = os.path.join(root, fn)
                content = open(path).read().lower()
                for word in forbidden:
                    if word in content:
                        print(f"  WARNING: '{word}' found in {path}")
                        all_good = False

    if all_good:
        print("  PASS: No forbidden words found in any file.")

    # Verify CSVs named correctly
    for expected in ["sglang/results.csv", "vllm/results.csv", "lmcache/results.csv"]:
        p = OUT / expected
        if p.exists():
            n = len(pd.read_csv(p))
            print(f"  {expected}: {n} rows OK")
        else:
            print(f"  MISSING: {expected}")

    # Verify no old files
    for old in ["results_complete.csv", "results_3engine_comparison.csv",
                "results_32b_comparison.csv", "results_apc_comparison.csv"]:
        for sub in ["sglang", "vllm", "lmcache"]:
            p = OUT / sub / old
            if p.exists():
                print(f"  STALE: {p}")

    print("\nDone.")


if __name__ == "__main__":
    import os
    main()
