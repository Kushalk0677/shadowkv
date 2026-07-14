from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json
from proactive_kv_cache.engines import (
    NoCacheEngine,
    ReactivePrefixCacheEngine,
    StrictReactivePrefixCacheEngine,
    compare_named_runs,
    maybe_shutdown,
)
from proactive_kv_cache.models import load_backend
from proactive_kv_cache.workload import make_synthetic_workload


def main() -> None:
    backend = load_backend('fake', device='cpu')
    requests = make_synthetic_workload('high_skew', n_requests=20, seed=11)
    engines = [
        NoCacheEngine(backend=backend, max_memory_mb=8),
        ReactivePrefixCacheEngine(backend=backend, max_memory_mb=8),
        StrictReactivePrefixCacheEngine(backend=backend, max_memory_mb=8),
    ]
    for engine in engines:
        for req in requests:
            tokens = backend.tokenize(req.prompt)
            engine.serve_tokens(req.request_id, tokens)
        maybe_shutdown(engine)

    summary = compare_named_runs(engines)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    out = results_dir / 'smoke_test_results.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
