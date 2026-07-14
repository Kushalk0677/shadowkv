from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset

from proactive_kv_cache import datasets as dataset_mod


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"


def _find_arrow_file(dataset_name: str, split: str, cache_root: Path) -> Path:
    cfg = dataset_mod.DATASET_REGISTRY[dataset_name]
    hf_name = str(cfg["hf_name"])
    cache_name = hf_name.replace("/", "___")
    candidates = sorted((cache_root / cache_name).glob(f"**/*-{split}.arrow"))
    if not candidates:
        candidates = sorted((cache_root / cache_name).glob(f"**/*{split}*.arrow"))
    if not candidates:
        raise FileNotFoundError(f"No cached Arrow split found for {dataset_name} split={split} under {cache_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_cached_loader(cache_root: Path):
    def load_public_text_rows(dataset_name: str, split: str, limit: int, seed: int = 42, prompt_mode: str = "raw"):
        if limit <= 0:
            raise ValueError("limit must be positive")
        if dataset_name not in dataset_mod.DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        cfg = dataset_mod.DATASET_REGISTRY[dataset_name]
        dataset_type = str(cfg["type"])
        actual_split = split or str(cfg["default_split"])
        resolved_prompt_mode = dataset_mod._resolve_prompt_mode(prompt_mode)
        arrow_file = _find_arrow_file(dataset_name, actual_split, cache_root)
        ds = Dataset.from_file(str(arrow_file))
        if hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=seed)

        rows = []
        for ex in ds:
            base_prompt = dataset_mod._row_to_prompt(dataset_type, ex)
            if not base_prompt:
                continue
            prompt, shared_prefix_text, mode_metadata = dataset_mod._apply_prompt_mode(
                dataset_name,
                dataset_type,
                base_prompt,
                resolved_prompt_mode,
                request_index=len(rows),
            )
            row = {
                "prompt": prompt,
                "source_dataset": dataset_name,
                "hf_name": cfg["hf_name"],
                "dataset_type": dataset_type,
                "prompt_mode": resolved_prompt_mode,
                "shared_prefix_text": shared_prefix_text,
                "cached_arrow_file": str(arrow_file),
            }
            row.update(mode_metadata)
            rows.append(row)
            if len(rows) >= limit:
                break
        if not rows:
            raise RuntimeError(f"No usable rows found in cached {dataset_name} split={actual_split}")
        return rows

    return load_public_text_rows


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    known, remaining = parser.parse_known_args()

    dataset_mod.load_public_text_rows = _build_cached_loader(Path(known.cache_root))
    sys.argv = [str(ROOT / "experiments" / "run_benchmark.py"), *remaining]

    from experiments.run_benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
