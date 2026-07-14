#!/usr/bin/env python3
"""
Profile ShadowKV+ to identify performance bottlenecks.

This script runs ShadowKV+ with profiling enabled to identify the most
time-consuming functions and operations.
"""

import cProfile
import pstats
import io
import sys
import time
from typing import Tuple, Dict

# Add the src directory to the path so we can import the modules
sys.path.insert(0, 'C:\\shadowkv\\robust_policy_upgrade_v6\\src')

from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import Backend, FakeBackend

def create_test_workload() -> list[tuple[Tuple[int, ...], Dict]]:
    """Create a synthetic workload for profiling."""
    # Generate a mix of request patterns to simulate real workload
    workload = []
    
    # Common prefix patterns (templated/rag)
    common_prefix = tuple(range(100, 150))  # 50 tokens
    for i in range(20):
        suffix = tuple(range(150 + i*10, 150 + i*10 + 30))
        tokens = common_prefix + suffix
        metadata = {
            'prompt_mode': 'templated',
            'shared_prefix_hint_tokens': len(common_prefix),
            'arrival_time': time.time() + i * 0.1
        }
        workload.append((tokens, metadata))
    
    # Some raw prompts
    for i in range(10):
        tokens = tuple(range(200 + i*20, 200 + i*20 + 80))
        metadata = {
            'prompt_mode': 'raw',
            'arrival_time': time.time() + (20 + i) * 0.1
        }
        workload.append((tokens, metadata))
    
    # Semantic patterns
    semantic_base = tuple(range(300, 380))  # 80 tokens
    for i in range(15):
        # Create paraphrased versions by modifying some tokens
        modified = list(semantic_base)
        if i % 3 == 0:
            modified[10:15] = [x + 1000 for x in modified[10:15]]  # Modify middle
        elif i % 3 == 1:
            modified[-10:] = [x + 2000 for x in modified[-10:]]  # Modify end
        else:
            modified[5:10] = [x + 3000 for x in modified[5:10]]  # Modify beginning
            
        tokens = tuple(modified)
        metadata = {
            'prompt_mode': 'semantic',
            'semantic_equivalence_key': f'semantic_family_{i // 5}',
            'arrival_time': time.time() + (30 + i) * 0.1
        }
        workload.append((tokens, metadata))
    
    return workload

def run_shadowkv_plus_with_profiling():
    """Run ShadowKV+ workload with profiling enabled."""
    print("Creating ShadowKV+ engine...")
    
    # Create backend and engine
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(
        backend=backend,
        max_memory_mb=256,
        semantic_similarity_threshold=0.58,
        allow_approximate_semantic_reuse=True
    )
    
    # Create workload
    workload = create_test_workload()
    print(f"Generated workload with {len(workload)} requests")
    
    # Run with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Execute workload
        start_time = time.time()
        for request_id, (tokens, metadata) in enumerate(workload):
            result = engine.serve_tokens(request_id, tokens, metadata)
            
            # Print progress every 10 requests
            if request_id % 10 == 0:
                print(f"Processed {request_id + 1}/{len(workload)} requests...")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nCompleted {len(workload)} requests in {total_time:.2f} seconds")
        print(f"Average latency: {total_time / len(workload) * 1000:.2f} ms")
        
    finally:
        profiler.disable()
        
        # Shutdown engine
        if hasattr(engine, 'shutdown'):
            engine.shutdown()
        else:
            engine.finalize()
    
    # Generate profiling report
    print("\n" + "="*60)
    print("PROFILING RESULTS - Top 30 Time-Consuming Functions")
    print("="*60)
    
    # Sort by cumulative time to see where most time is spent
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print(s.getvalue())
    
    # Also show functions with highest internal time
    print("\n" + "="*60)
    print("PROFILING RESULTS - Top 20 Functions by Internal Time")
    print("="*60)
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('time')
    ps2.print_stats(20)  # Top 20 functions
    
    print(s2.getvalue())
    
    # Save detailed profiling data to file
    profiler.dump_stats('shadowkv_plus_profile.prof')
    print(f"\nDetailed profiling data saved to: shadowkv_plus_profile.prof")
    
    # Generate a more detailed report file
    with open('shadowkv_plus_profile.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats()
        print("\n" + "="*60, file=f)
        print("Callers for key functions:", file=f)
        print("="*60, file=f)
        
        # Show callers for key ShadowKV+ methods
        key_functions = [
            '_plan_for_request',
            '_semantic_best_match',
            '_partial_semantic_reuse',
            'serve_tokens',
            '_observe_request',
            '_store_reactive_prefix',
            '_should_attempt_cache_use'
        ]
        
        for func_name in key_functions:
            print(f"\nCallers for {func_name}:", file=f)
            print("-" * 40, file=f)
            try:
                ps.print_callers(func_name)
            except:
                print(f"No data for {func_name}", file=f)
    
    print(f"Detailed profiling report saved to: shadowkv_plus_profile.txt")

if __name__ == "__main__":
    print("ShadowKV+ Profiling Tool")
    print("=" * 40)
    print("This tool will run ShadowKV+ with a synthetic workload")
    print("and generate detailed profiling information.")
    print()
    
    run_shadowkv_plus_with_profiling()
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
    print("Key files generated:")
    print("  - shadowkv_plus_profile.prof (binary profiling data)")
    print("  - shadowkv_plus_profile.txt (detailed text report)")
    print()
    print("Next steps:")
    print("1. Analyze the profiling results to identify bottlenecks")
    print("2. Focus on functions with high cumulative time")
    print("3. Look for opportunities to optimize or cache expensive operations")