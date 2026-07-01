"""Profile Plan() latency: 10,000 synthetic requests, no GPU needed."""
import sys, os, time, json
sys.path.insert(0, os.path.join('v10', 'src'))

from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import FakeBackend

backend = FakeBackend()
engine = ShadowKVPlusEngine(backend=backend, max_memory_mb=256)

# Generate synthetic inputs: mix of token lengths and prompt modes
import random
random.seed(42)

latencies = []
n = 1000

for i in range(n):
    # Vary token length: 10-400 tokens
    token_len = random.randint(10, 400)
    tokens = tuple(random.randint(0, 50000) for _ in range(token_len))
    
    # Vary prompt mode: 40% raw, 40% templated, 20% semantic
    mode = random.choices(
        ['raw', 'templated', 'semantic'],
        weights=[0.4, 0.4, 0.2]
    )[0]
    
    metadata = {
        'prompt_mode': mode,
        'arrival_time': time.time() + i * 0.001,
    }
    if mode == 'templated':
        metadata['shared_prefix_hint_tokens'] = token_len // 2
        metadata['shared_prefix_text'] = ' '.join(f'word{j}' for j in range(10))
    
    t0 = time.perf_counter()
    result = engine.serve_tokens(i, tokens, metadata)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)  # ms

# Read accumulated metrics
calls = engine.engine_metrics.get('policy_planning_calls', 0)
total_ms = engine.engine_metrics.get('policy_planning_latency_total_ms', 0.0)
sem_calls = engine.engine_metrics.get('semantic_query_calls', 0)
sem_total_ms = engine.engine_metrics.get('semantic_query_latency_total_ms', 0.0)

print(f'=== Plan() Profiling: {n} requests ===')
print(f'Policy planning calls: {calls}')
print(f'Policy planning total: {total_ms:.3f} ms')
print(f'Policy planning mean:  {total_ms/calls:.4f} ms' if calls else 'N/A')
print()
print(f'Semantic query calls:  {sem_calls}')
print(f'Semantic query total:  {sem_total_ms:.3f} ms')
print(f'Semantic query mean:   {sem_total_ms/sem_calls:.4f} ms' if sem_calls else 'N/A')
print()

# Per-call distribution from raw latencies
latencies.sort()
mean_lat = sum(latencies) / len(latencies)
p50 = latencies[len(latencies)//2]
p95 = latencies[int(len(latencies)*0.95)]
p99 = latencies[int(len(latencies)*0.99)]
p999 = latencies[int(len(latencies)*0.999)]
print(f'End-to-end serve_tokens() latency (includes backend ops):')
print(f'  Mean: {mean_lat:.4f} ms')
print(f'  P50:  {p50:.4f} ms')
print(f'  P95:  {p95:.4f} ms')
print(f'  P99:  {p99:.4f} ms')
print(f'  P999: {p999:.4f} ms')
print(f'  Max:  {latencies[-1]:.4f} ms')
