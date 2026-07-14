"""Verify coupling penalty is applied in both admission paths."""
import sys, os
sys.path.insert(0, os.path.join('v10', 'src'))
import inspect
from proactive_kv_cache.utility import UtilityModel

# Check exact path
src_exact = inspect.getsource(UtilityModel._score_exact_prefix)
has_coupling_exact = 'coupling_penalty' in src_exact and 'lambda_risk' in src_exact

# Check semantic path
src_sem = inspect.getsource(UtilityModel._score_semantic_prefix)
has_coupling_sem = 'coupling' in src_sem and 'lambda_risk' in src_sem

print(f'Exact path has coupling penalty:    {has_coupling_exact}')
print(f'Semantic path has coupling penalty: {has_coupling_sem}')

if has_coupling_exact and has_coupling_sem:
    print('ALL OK - both paths include risk-averse coupling')
else:
    print('ISSUE FOUND')
    sys.exit(1)
