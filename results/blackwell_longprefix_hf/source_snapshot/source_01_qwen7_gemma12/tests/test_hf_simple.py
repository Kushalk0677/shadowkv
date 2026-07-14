#!/usr/bin/env python3
"""
Simple test to verify HuggingFace backend works with our changes
"""

import sys
sys.path.insert(0, 'src')

try:
    from proactive_kv_cache.models import load_backend
    
    # Try loading a small model
    print("Attempting to load a small model...")
    backend = load_backend("hf", model_name="gpt2", device="cpu", dtype="auto")
    
    # Test tokenization
    tokens = backend.tokenize("Hello world")
    print(f"Tokenization successful: {len(tokens)} tokens")
    
    # Test prefill
    result = backend.prefill(tokens)
    print(f"Prefill successful: {result.latency_ms}ms")
    
    print("✅ HuggingFace backend test passed!")
    
except Exception as e:
    print(f"❌ HuggingFace backend test failed: {e}")
    import traceback
    traceback.print_exc()