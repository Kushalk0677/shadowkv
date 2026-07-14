"""Check the KV cache format in transformers 5.x."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
m = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

x = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    o = m(x, use_cache=True)
pk = o.past_key_values

print(f'Type: {type(pk).__name__}')
print(f'Len: {len(pk)}')
print(f'Elem type: {type(pk[0]).__name__}')
print(f'Elem len: {len(pk[0])}')

# Check first element
first = pk[0]
for i, kv in enumerate(first):
    print(f'  [{i}] type={type(kv).__name__}, shape={kv.shape}, dtype={kv.dtype}')

# Check DynamicCache methods
if hasattr(pk, 'get_seq_length'):
    print(f'get_seq_length(): {pk.get_seq_length()}')
if hasattr(pk, 'to_legacy_cache'):
    lc = pk.to_legacy_cache()
    print(f'Legacy type: {type(lc).__name__}')
    print(f'Legacy len: {len(lc)}')
    first_lc = lc[0]
    for i, kv in enumerate(first_lc):
        print(f'  Legacy[{i}]: type={type(kv).__name__}, shape={kv.shape}')
