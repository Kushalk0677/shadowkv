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
print(f'get_seq_length(): {pk.get_seq_length()}')
print(f'key_cache len: {len(pk.key_cache)}')
print(f'value_cache len: {len(pk.value_cache)}')
print(f'key_cache[0] shape: {pk.key_cache[0].shape}')
print(f'value_cache[0] shape: {pk.value_cache[0].shape}')

# Try slicing
sliced = DynamicCache()
for i in range(len(pk.key_cache)):
    sliced.key_cache.append(pk.key_cache[i][:, :, :3, :])
    sliced.value_cache.append(pk.value_cache[i][:, :, :3, :])
print(f'Sliced seq length: {sliced.get_seq_length()}')

# Try using sliced cache
suffix = torch.tensor([[6, 7]])
with torch.no_grad():
    o2 = m(suffix, past_key_values=sliced, use_cache=True)
print(f'After append, seq length: {o2.past_key_values.get_seq_length()}')
print('KV reuse test: PASSED')
