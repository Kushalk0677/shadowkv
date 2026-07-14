"""Check DynamicCache layer structure."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

t = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
m = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

x = torch.tensor([[101, 102, 103, 104, 105]])
with torch.no_grad():
    o = m(x, use_cache=True)
pk = o.past_key_values

print(f'Num layers: {len(pk.layers)}')
layer0 = pk.layers[0]
print(f'Layer type: {type(layer0).__name__}')
print(f'Layer attrs: {[a for a in dir(layer0) if not a.startswith("_")]}')

# Check key_cache and value_cache
if hasattr(layer0, 'key_cache'):
    print(f'key_cache shape: {layer0.key_cache.shape}')
if hasattr(layer0, 'value_cache'):
    print(f'value_cache shape: {layer0.value_cache.shape}')

# Try to see what the layer object stores
print(f'layer0 keys: {layer0.__dict__.keys() if hasattr(layer0, "__dict__") else "no __dict__"}')

# Try slicing via crop method (common in newer DynamicCache)
if hasattr(pk, 'crop'):
    pk.crop(3)
    print(f'After crop(3): seq_len={pk.get_seq_length()}')

# Test with suffix
suffix = torch.tensor([[106, 107]])
try:
    o2 = m(suffix, past_key_values=pk, use_cache=True)
    print(f'Cache append worked! New seq: {o2.past_key_values.get_seq_length()}')
except Exception as e:
    print(f'Cache append failed: {e}')
