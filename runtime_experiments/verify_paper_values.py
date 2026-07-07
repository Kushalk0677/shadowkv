#!/usr/bin/env python3
"""
verify_paper_values.py — Verify all CSV values match the paper's claimed numbers.
"""
import pandas as pd
import sys

SGLANG_CSV = r"C:\Users\kusha\Documents\GitHub\shadowkv\runtime_experiments\sglang\results.csv"
VLLM_CSV = r"C:\Users\kusha\Documents\GitHub\shadowkv\runtime_experiments\vllm\results.csv"

errors = []
def check(desc, ok):
    if ok:
        print(f"  PASS: {desc}")
    else:
        print(f"  FAIL: {desc}")
        errors.append(desc)

# ── SGLang checks ──────────────────────────────────────────────────────
s = pd.read_csv(SGLANG_CSV)
# Determine column name for model
s_model_key = 'model_slug'

# 7B LMCache mean = 28.6ms
lm7 = s[(s[s_model_key]=='qwen25_7b')&(s['engine']=='lmcache_no_native_radix')]['mean_latency_ms'].mean()
check("SGLang 7B LMCache mean = 28.6ms", abs(lm7 - 28.6) < 0.2)

# 7B ShadowKV++ speedup vs LMCache = +16.7%
sk7 = s[(s[s_model_key]=='qwen25_7b')&(s['engine']=='sglang_radix_attention_shadowkv_plus')]
sp7 = sk7['speedup_vs_lmcache_pct'].mean()
check("SGLang 7B SKVPP speedup = +16.7%", abs(sp7 - 16.7) < 0.5)

# ── vLLM 32B checks ───────────────────────────────────────────────────
v = pd.read_csv(VLLM_CSV)
v_model_key = 'model_slug'

# 32B no-cache mean = 72.8ms
nc32 = v[(v[v_model_key]=='qwen25_32b')&(v['engine']=='vllm_no_cache')]['mean_latency_ms'].mean()
check("vLLM 32B no-cache mean = 72.8ms", abs(nc32 - 72.8) < 0.2)

# 32B APC mean = 59.4ms
apc32 = v[(v[v_model_key]=='qwen25_32b')&(v['engine']=='vllm_apc')]['mean_latency_ms'].mean()
check("vLLM 32B APC mean = 59.4ms", abs(apc32 - 59.4) < 0.2)

# 32B APC+ mean = 59.0ms
sk32 = v[(v[v_model_key]=='qwen25_32b')&(v['engine']=='vllm_apc_shadowkv_plus')]['mean_latency_ms'].mean()
check("vLLM 32B APC+ mean = 59.0ms", abs(sk32 - 59.0) < 0.2)

# 32B speedups (ratio of means)
sp_apc = (nc32 - apc32) / nc32 * 100
sp_sk = (nc32 - sk32) / nc32 * 100
check("vLLM 32B APC speedup = 18.4%", abs(sp_apc - 18.4) < 0.2)
check("vLLM 32B APC+ speedup = 19.0%", abs(sp_sk - 19.0) < 0.2)

# 32B Energy sums
for eng, label, expected in [
    ('vllm_no_cache', 'no-cache', 111.3),
    ('vllm_apc', 'APC', 83.4),
    ('vllm_apc_shadowkv_plus', 'APC+', 83.6),
]:
    eng_e = v[(v[v_model_key]=='qwen25_32b')&(v['engine']==eng)]['gpu_energy_j'].sum()
    check(f"vLLM 32B {label} energy = {expected}KJ", abs(eng_e/1000 - expected) < 1.0)

# ── vLLM 7B checks ────────────────────────────────────────────────────
nc7 = v[(v[v_model_key]=='qwen25_7b')&(v['engine']=='vllm_no_cache')]['mean_latency_ms'].mean()
check("vLLM 7B no-cache mean = 35.9ms", abs(nc7 - 35.9) < 0.2)

apc7 = v[(v[v_model_key]=='qwen25_7b')&(v['engine']=='vllm_apc')]['mean_latency_ms'].mean()
check("vLLM 7B APC mean = 27.7ms", abs(apc7 - 27.7) < 0.2)

sk7v = v[(v[v_model_key]=='qwen25_7b')&(v['engine']=='vllm_apc_shadowkv_plus')]['mean_latency_ms'].mean()
check("vLLM 7B APC+ mean = 27.1ms", abs(sk7v - 27.1) < 0.2)

sp_apc7 = (nc7 - apc7) / nc7 * 100
sp_sk7 = (nc7 - sk7v) / nc7 * 100
check("vLLM 7B APC speedup = 22.8%", abs(sp_apc7 - 22.8) < 0.5)
check("vLLM 7B APC+ speedup = 24.5%", abs(sp_sk7 - 24.5) < 0.5)

print()
if errors:
    print(f"{len(errors)} FAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All checks passed!")
