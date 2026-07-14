"""Fix Qwen by setting explicit attention mask."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])
attn = torch.ones_like(ids)  # all tokens are real (no padding)

# Clean
with torch.no_grad():
    out_clean = m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids.shape[1]:], skip_special_tokens=True)
print(f'Clean: {repr(clean_text)}')

# Prefill + generate with attention mask
with torch.no_grad():
    out = m(input_ids=ids, attention_mask=attn, use_cache=True)
cache = out.past_key_values
cache.crop(ids.shape[1] - 1)
last_tok = ids[:, -1:]
last_attn = torch.ones((1, 1))

# Try with explicit attention mask in the manual loop
gen_ids = []
inp = last_tok
inp_attn = last_attn
kv = cache
prev_attn = attn[:, :-1]  # attention mask for cached positions

for step in range(16):
    # Concatenate previous attention mask with current
    full_attn = torch.cat([prev_attn, inp_attn], dim=1)
    with torch.no_grad():
        outs = m(input_ids=inp, attention_mask=full_attn, past_key_values=kv, use_cache=True)
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    inp = nid
    inp_attn = torch.ones((1, 1))
    kv = outs.past_key_values
    prev_attn = full_attn
    gen_ids.append(nid.item())
    if nid.item() == tok.eos_token_id:
        break

reuse_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'Reuse: {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
