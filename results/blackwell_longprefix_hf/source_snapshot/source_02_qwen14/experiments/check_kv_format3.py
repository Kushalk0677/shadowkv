"""Check DynamicCache API."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
m = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

x = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    o = m(x, use_cache=True)
pk = o.past_key_values

print(f'Type: {type(pk).__name__}')
attrs = [a for a in dir(pk) if not a.startswith('_')]
print(f'Public attrs: {attrs}')
print(f'get_seq_length(): {pk.get_seq_length()}')

# Try to_legacy_cache
if hasattr(pk, 'to_legacy_cache'):
    lc = pk.to_legacy_cache()
    print(f'Legacy type: {type(lc).__name__}')
    print(f'Legacy len: {len(lc)}')
    first = lc[0]
    print(f'First layer: {len(first)} tensors')
    for i, kv in enumerate(first):
        print(f'  [{i}] shape={kv.shape}')

# Try using legacy cache as past_key_values
suffix = torch.tensor([[6, 7]])
try:
    o2 = m(suffix, past_key_values=lc, use_cache=True)
    print(f'Legacy cache worked! New seq: {o2.past_key_values.get_seq_length()}')
except Exception as e:
    print(f'Legacy cache failed: {e}')
    # Try wrapping in DynamicCache
    from transformers.cache_utils import DynamicCache
    try:
        dc = DynamicCache.from_legacy_cache(lc)
        print(f'DynamicCache from legacy: seq={dc.get_seq_length()}')
        # Slice it
        sliced_tensors = []
        for i in range(len(lc)):
            k, v = lc[i]
            sliced_tensors.append((k[:, :, :3, :], v[:, :, :3, :]))
        sliced_lc = tuple(sliced_tensors)
        o3 = m(suffix, past_key_values=sliced_lc, use_cache=True)
        print(f'Sliced legacy cache worked! New seq: {o3.past_key_values.get_seq_length()}')
    except Exception as e2:
        print(f'Also failed: {e2}')
