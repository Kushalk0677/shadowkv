"""Debug: Why does Qwen fail KV reuse test?"""
import warnings, torch, json
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

# Check Qwen's chat template
print(f'Chat template: {tok.chat_template[:100] if tok.chat_template else "None"}')

# Test with identical prompts (100% shared)
prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."

ids = tok.encode(prompt, truncation=True, max_length=384)
ids_t = torch.tensor([ids])

# Clean generation
with torch.no_grad():
    out_clean = m.generate(ids_t, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids_t.shape[1]:], skip_special_tokens=True)
print(f'Clean: {repr(clean_text)}')

# KV reuse with 100% shared prefix
with torch.no_grad():
    out = m(ids_t, use_cache=True)
cache = out.past_key_values

shared = ids_t.shape[1]
cache.crop(shared)
last_tok = ids_t[:, -1:]

# Manual generation
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
