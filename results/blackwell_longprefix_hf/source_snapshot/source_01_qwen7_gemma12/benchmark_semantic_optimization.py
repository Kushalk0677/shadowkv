#!/usr/bin/env python3
"""
Benchmark the semantic analysis optimization in ShadowKV+.

This script compares the performance before and after the optimization
to quantify the speedup.
"""

import sys
import time
import copy
from typing import Tuple, Dict

# Add the src directory to the path
sys.path.insert(0, 'C:\\shadowkv\\robust_policy_upgrade_v6\\src')

from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import Backend, FakeBackend

def create_workload():
    """Create a mixed workload for benchmarking."""
    workload = []
    
    # 1. Short requests (should skip semantic analysis)
    for i in range(10):
        tokens = tuple(range(100 + i*5, 100 + i*5 + 20))  # 20 tokens
        metadata = {'prompt_mode': 'raw', 'arrival_time': time.time() + i * 0.01}
        workload.append((tokens, metadata, 'short_raw'))
    
    # 2. Raw prompts with low reuse (should skip semantic analysis)
    for i in range(15):
        tokens = tuple(range(200 + i*10, 200 + i*10 + 40))  # 40 tokens
        metadata = {'prompt_mode': 'raw', 'arrival_time': time.time() + (10 + i) * 0.01}
        workload.append((tokens, metadata, 'long_raw_low_reuse'))
    
    # 3. Templated prompts (should use semantic analysis)
    common_prefix = tuple(range(300, 350))  # 50 tokens
    for i in range(12):
        suffix = tuple(range(350 + i*5, 350 + i*5 + 25))
        tokens = common_prefix + suffix
        metadata = {
            'prompt_mode': 'templated',
            'shared_prefix_hint_tokens': len(common_prefix),
            'arrival_time': time.time() + (25 + i) * 0.01
        }
        workload.append((tokens, metadata, 'templated_high_reuse'))
    
    # 4. Semantic prompts (should use semantic analysis)
    semantic_base = tuple(range(400, 460))  # 60 tokens
    for i in range(8):
        modified = list(semantic_base)
        if i % 2 == 0:
            modified[15:20] = [x + 1000 for x in modified[15:20]]
        tokens = tuple(modified)
        metadata = {
            'prompt_mode': 'semantic',
            'semantic_equivalence_key': f'family_{i // 4}',
            'arrival_time': time.time() + (37 + i) * 0.01
        }
        workload.append((tokens, metadata, 'semantic'))
    
    return workload

def run_benchmark(workload, description):
    """Run a benchmark with the given workload."""
    print(f"\n{description}")
    print("-" * 60)
    
    # Create engine
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(
        backend=backend,
        max_memory_mb=256,
        semantic_similarity_threshold=0.58,
        allow_approximate_semantic_reuse=True
    )
    
    # Run workload
    start_time = time.time()
    
    semantic_queries = 0
    semantic_skipped = 0
    
    for request_id, (tokens, metadata, category) in enumerate(workload):
        result = engine.serve_tokens(request_id, tokens, metadata)
        
        # Track semantic analysis usage
        if request_id % 5 == 0:  # Sample every 5th request to avoid overhead
            should_use = engine._should_use_semantic_analysis(tokens, metadata)
            if should_use:
                semantic_queries += 1
            else:
                semantic_skipped += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Cleanup
    if hasattr(engine, 'shutdown'):
        engine.shutdown()
    else:
        engine.finalize()
    
    # Get metrics
    total_requests = len(workload)
    avg_latency = total_time / total_requests * 1000  # ms
    
    semantic_queries_total = engine.engine_metrics.get('semantic_queries_total', 0)
    semantic_skipped_total = engine.engine_metrics.get('semantic_queries_skipped_total', 0)
    
    print(f"  Total requests: {total_requests}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Semantic queries executed: {semantic_queries_total}")
    print(f"  Semantic queries skipped: {semantic_skipped_total}")
    print(f"  Semantic analysis rate: {(semantic_queries_total / max(total_requests, 1)):.1%}")
    
    return {
        'total_time': total_time,
        'avg_latency': avg_latency,
        'total_requests': total_requests,
        'semantic_queries': semantic_queries_total,
        'semantic_skipped': semantic_skipped_total,
        'engine': engine
    }

def main():
    print("ShadowKV+ Semantic Optimization Benchmark")
    print("=" * 60)
    print("This benchmark compares performance with selective semantic analysis")
    print("vs. always-running semantic analysis (simulated by forcing it on).")
    
    # Create workload
    workload = create_workload()
    print(f"\nGenerated workload with {len(workload)} requests:")
    
    # Count request types
    categories = {}
    for _, _, category in workload:
        categories[category] = categories.get(category, 0) + 1
    
    for category, count in categories.items():
        print(f"  {category}: {count} requests")
    
    # Run optimized version
    optimized_results = run_benchmark(workload, "OPTIMIZED (Selective Semantic Analysis)")
    
    # For comparison, let's simulate the "before" scenario by temporarily disabling the optimization
    # We'll do this by patching the _should_use_semantic_analysis method to always return True
    print("\n" + "=" * 60)
    
    # Create a version that always uses semantic analysis (simulating pre-optimization)
    class AlwaysSemanticShadowKVPlusEngine(ShadowKVPlusEngine):
        def _should_use_semantic_analysis(self, tokens, metadata):
            # Always use semantic analysis (simulates pre-optimization behavior)
            return True
    
    # Run "before" version
    backend_before = FakeBackend()
    engine_before = AlwaysSemanticShadowKVPlusEngine(
        backend=backend_before,
        max_memory_mb=256,
        semantic_similarity_threshold=0.58,
        allow_approximate_semantic_reuse=True
    )
    
    print("BEFORE (Always Semantic Analysis - Simulated)")
    print("-" * 60)
    
    start_time = time.time()
    semantic_queries_before = 0
    
    for request_id, (tokens, metadata, category) in enumerate(workload):
        result = engine_before.serve_tokens(request_id, tokens, metadata)
        
        # Since we're forcing semantic analysis, count all requests
        if request_id % 5 == 0:
            semantic_queries_before += 1
    
    end_time = time.time()
    total_time_before = end_time - start_time
    
    if hasattr(engine_before, 'shutdown'):
        engine_before.shutdown()
    else:
        engine_before.finalize()
    
    total_requests = len(workload)
    avg_latency_before = total_time_before / total_requests * 1000
    
    semantic_queries_total_before = engine_before.engine_metrics.get('semantic_queries_total', 0)
    semantic_skipped_total_before = engine_before.engine_metrics.get('semantic_queries_skipped_total', 0)
    
    print(f"  Total requests: {total_requests}")
    print(f"  Total time: {total_time_before:.3f} seconds")
    print(f"  Average latency: {avg_latency_before:.2f} ms")
    print(f"  Semantic queries executed: {semantic_queries_total_before}")
    print(f"  Semantic queries skipped: {semantic_skipped_total_before}")
    print(f"  Semantic analysis rate: {(semantic_queries_total_before / max(total_requests, 1)):.1%}")
    
    before_results = {
        'total_time': total_time_before,
        'avg_latency': avg_latency_before,
        'total_requests': total_requests,
        'semantic_queries': semantic_queries_total_before,
        'semantic_skipped': semantic_skipped_total_before
    }
    
    # Calculate improvement
    print("\n" + "=" * 60)
    print("PERFORMANCE IMPROVEMENT")
    print("-" * 60)
    
    time_reduction = ((before_results['total_time'] - optimized_results['total_time']) / before_results['total_time']) * 100
    latency_reduction = ((before_results['avg_latency'] - optimized_results['avg_latency']) / before_results['avg_latency']) * 100
    semantic_reduction = ((before_results['semantic_queries'] - optimized_results['semantic_queries']) / max(before_results['semantic_queries'], 1)) * 100
    
    print(f"  Time reduction: {time_reduction:.1f}%")
    print(f"  Latency reduction: {latency_reduction:.1f}%")
    print(f"  Semantic queries reduction: {semantic_reduction:.1f}%")
    print(f"  Speedup: {before_results['avg_latency'] / optimized_results['avg_latency']:.2f}x")
    
    print(f"\nSUMMARY:")
    print(f"  Before: {before_results['avg_latency']:.2f} ms avg latency, {before_results['semantic_queries']} semantic queries")
    print(f"  After:  {optimized_results['avg_latency']:.2f} ms avg latency, {optimized_results['semantic_queries']} semantic queries")
    print(f"  Improvement: {latency_reduction:.1f}% faster, {semantic_reduction:.1f}% fewer semantic queries")
    
    # Show which request types benefited most
    print(f"\nREQUEST TYPE BREAKDOWN:")
    print(f"  Short raw requests (< 32 tokens): {categories.get('short_raw', 0)} requests - SKIP semantic")
    print(f"  Long raw low-reuse requests: {categories.get('long_raw_low_reuse', 0)} requests - SKIP semantic")
    print(f"  Templated high-reuse requests: {categories.get('templated_high_reuse', 0)} requests - USE semantic")
    print(f"  Semantic requests: {categories.get('semantic', 0)} requests - USE semantic")

if __name__ == "__main__":
    main()