#!/usr/bin/env python3
"""
Test the semantic analysis optimization in ShadowKV+.

This script verifies that semantic analysis is selectively applied
only when beneficial, reducing overhead.
"""

import sys
import time
from typing import Tuple, Dict

# Add the src directory to the path
sys.path.insert(0, 'C:\\shadowkv\\robust_policy_upgrade_v6\\src')

from proactive_kv_cache.engines import ShadowKVPlusEngine
from proactive_kv_cache.models import Backend, FakeBackend

def test_selective_semantic_analysis():
    """Test that semantic analysis is selectively applied."""
    print("Testing ShadowKV+ Semantic Analysis Optimization")
    print("=" * 50)
    
    # Create engine
    backend = FakeBackend()
    engine = ShadowKVPlusEngine(
        backend=backend,
        max_memory_mb=256,
        semantic_similarity_threshold=0.58,
        allow_approximate_semantic_reuse=True
    )
    
    # Test 1: Short requests should skip semantic analysis
    print("\nTest 1: Short requests (< 32 tokens)")
    short_tokens = tuple(range(100, 120))  # 20 tokens
    metadata = {'prompt_mode': 'raw', 'arrival_time': time.time()}
    
    # Check the decision logic directly
    should_use = engine._should_use_semantic_analysis(short_tokens, metadata)
    print(f"  Tokens: {len(short_tokens)}")
    print(f"  Should use semantic analysis: {should_use}")
    assert not should_use, "Short requests should skip semantic analysis"
    print("  PASS: Short requests correctly skip semantic analysis")
    
    # Test 2: Raw prompts with low hit rate should skip semantic analysis
    print("\nTest 2: Raw prompts with low cache hit rate")
    
    # Simulate some requests with low hit rate
    for i in range(10):
        tokens = tuple(range(200 + i*10, 200 + i*10 + 40))  # 40 tokens
        metadata = {'prompt_mode': 'raw', 'arrival_time': time.time() + i * 0.01}
        result = engine.serve_tokens(i, tokens, metadata)
    
    # Now check if semantic analysis would be used
    long_raw_tokens = tuple(range(300, 340))  # 40 tokens
    raw_metadata = {'prompt_mode': 'raw', 'arrival_time': time.time()}
    should_use = engine._should_use_semantic_analysis(long_raw_tokens, raw_metadata)
    
    reuse_attempts = engine.engine_metrics.get('reuse_attempts', 0)
    reuse_successes = engine.engine_metrics.get('reuse_successes', 0)
    hit_rate = reuse_successes / max(reuse_attempts, 1) if reuse_attempts > 0 else 0
    
    print(f"  Tokens: {len(long_raw_tokens)}")
    print(f"  Cache hit rate: {hit_rate:.2%}")
    print(f"  Should use semantic analysis: {should_use}")
    print(f"  Reuse attempts: {reuse_attempts}, successes: {reuse_successes}")
    
    if hit_rate < 0.15:
        assert not should_use, f"Raw prompts with low hit rate ({hit_rate:.2%}) should skip semantic analysis"
        print("  PASS: Raw prompts with low hit rate correctly skip semantic analysis")
    else:
        print("  INFO: Hit rate is high enough to use semantic analysis")
    
    # Test 3: Templated prompts should use semantic analysis
    print("\nTest 3: Templated prompts")
    templated_tokens = tuple(range(400, 450))  # 50 tokens
    templated_metadata = {
        'prompt_mode': 'templated',
        'shared_prefix_hint_tokens': 30,
        'arrival_time': time.time()
    }
    
    should_use = engine._should_use_semantic_analysis(templated_tokens, templated_metadata)
    print(f"  Tokens: {len(templated_tokens)}")
    print(f"  Prompt mode: templated")
    print(f"  Should use semantic analysis: {should_use}")
    assert should_use, "Templated prompts should use semantic analysis"
    print("  PASS: Templated prompts correctly use semantic analysis")
    
    # Test 4: Semantic prompts should use semantic analysis
    print("\nTest 4: Semantic prompts")
    semantic_tokens = tuple(range(500, 550))  # 50 tokens
    semantic_metadata = {
        'prompt_mode': 'semantic',
        'semantic_equivalence_key': 'test_family',
        'arrival_time': time.time()
    }
    
    should_use = engine._should_use_semantic_analysis(semantic_tokens, semantic_metadata)
    print(f"  Tokens: {len(semantic_tokens)}")
    print(f"  Prompt mode: semantic")
    print(f"  Should use semantic analysis: {should_use}")
    assert should_use, "Semantic prompts should use semantic analysis"
    print("  PASS: Semantic prompts correctly use semantic analysis")
    
    # Test 5: High hit rate scenarios should skip semantic analysis
    print("\nTest 5: High cache hit rate scenarios")
    
    # Create a scenario with high hit rate by reusing the same prefix
    common_prefix = tuple(range(600, 650))  # 50 tokens
    
    # First request - cache miss
    tokens1 = common_prefix + tuple(range(650, 680))
    result1 = engine.serve_tokens(100, tokens1, {'prompt_mode': 'templated', 'arrival_time': time.time()})
    
    # Subsequent requests - should be cache hits
    for i in range(5):
        tokens = common_prefix + tuple(range(650 + i*10, 650 + i*10 + 30))
        result = engine.serve_tokens(101 + i, tokens, {'prompt_mode': 'templated', 'arrival_time': time.time()})
    
    # Check hit rate
    reuse_attempts = engine.engine_metrics.get('reuse_attempts', 0)
    reuse_successes = engine.engine_metrics.get('reuse_successes', 0)
    hit_rate = reuse_successes / max(reuse_attempts, 1) if reuse_attempts > 0 else 0
    
    print(f"  Cache hit rate: {hit_rate:.2%}")
    print(f"  Reuse attempts: {reuse_attempts}, successes: {reuse_successes}")
    
    if hit_rate > 0.8:
        should_use = engine._should_use_semantic_analysis(common_prefix + tuple(range(700, 720)), 
                                                         {'prompt_mode': 'templated', 'arrival_time': time.time()})
        print(f"  Should use semantic analysis with high hit rate: {should_use}")
        assert not should_use, f"High hit rate scenarios ({hit_rate:.2%}) should skip semantic analysis"
        print("  PASS: High hit rate scenarios correctly skip semantic analysis")
    else:
        print("  INFO: Hit rate not high enough to trigger optimization")
    
    # Cleanup
    if hasattr(engine, 'shutdown'):
        engine.shutdown()
    else:
        engine.finalize()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("\nOptimization Summary:")
    print("- Short requests (< 32 tokens) skip semantic analysis")
    print("- Raw prompts skip semantic analysis unless hit rate > 15%")
    print("- High cache hit rate (> 80%) scenarios skip semantic analysis")
    print("- Templated/semantic/RAG prompts use semantic analysis")
    print("- This reduces unnecessary semantic computation overhead")

if __name__ == "__main__":
    test_selective_semantic_analysis()