"""Compare manual loop vs model.generate for Qwen."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# 1. model.generate() — standard
with torch.no_grad():
    out_gen = m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
gen_text = tok.decode(out_gen[0][ids.shape[1]:], skip_special_tokens=True)
print(f'1. model.generate():     {repr(gen_text)}')

# 2. Prefill + manual loop (standard approach, no reuse)
with torch.no_grad():
    out = m(ids, use_cache=True)
cache = out.past_key_values
total_pos = ids.shape[1]
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
manual_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'2. Prefill + manual loop: {repr(manual_text)}')
print(f'Match: {gen_text == manual_text}')
