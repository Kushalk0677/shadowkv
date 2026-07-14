"""Debug: Why does KV reuse produce different outputs than the clean run?"""
import warnings, json
warnings.filterwarnings('ignore')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
m = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

# Load a sample
ds = json.load(open('v10/fidelity_equiv_v7/tinyllama_results.json'))
item = ds[0]
prompt = None  # We don't have the original prompt in the result

# Instead, construct a simple test: identical prompts (100% shared)
prompt_a = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
prompt_b = prompt_a  # exactly the same

ids_a = tok(prompt_a, return_tensors='pt').input_ids
ids_b = tok(prompt_b, return_tensors='pt').input_ids

# Method 1: Clean generation from B
with torch.no_grad():
    out_clean = m.generate(ids_b, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids_b.shape[1]:], skip_special_tokens=True)
print(f'Clean:      {repr(clean_text)}')

# Method 2: Reuse from A's KV (identical prompt, so 100% shared)
with torch.no_grad():
    out_a = m(ids_a, use_cache=True)
cache_a = out_a.past_key_values  # DynamicCache for full prompt A

shared = ids_a.shape[1]  # Full prompt is shared
cache_a.crop(shared)

# No suffix since prompts are identical
last_tok = ids_a[:, -1:]

# Manual generation
eos_id = tok.eos_token_id or 0
gen_ids = []
inp = last_tok
kv = cache_a

for _ in range(16):
    with torch.no_grad():
        outs = m(input_ids=inp, past_key_values=kv, use_cache=True)
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    inp = nid
    kv = outs.past_key_values
    gen_ids.append(nid.item())
    if nid.item() == eos_id:
        break

reuse_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'Reuse:      {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
