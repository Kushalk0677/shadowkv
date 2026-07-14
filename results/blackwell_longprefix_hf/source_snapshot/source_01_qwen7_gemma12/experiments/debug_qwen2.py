"""Debug Qwen: with proper crop(total_pos - 1)."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Clean generation
with torch.no_grad():
    out_clean = m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids.shape[1]:], skip_special_tokens=True)
print(f'Clean: {repr(clean_text)}')

# KV reuse with proper crop
with torch.no_grad():
    out = m(ids, use_cache=True)

# Crop to total_pos - 1 and pass last token
total_pos = ids.shape[1]
cache = out.past_key_values
cache.crop(total_pos - 1)
last_tok = ids[:, -1:]

eos_id = tok.eos_token_id or 0
gen_ids = []
inp = last_tok
kv = cache
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
print(f'Reuse: {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
