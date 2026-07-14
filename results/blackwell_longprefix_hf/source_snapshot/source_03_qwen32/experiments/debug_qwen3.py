"""Debug Qwen: use model.generate() with past_key_values."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Clean generation with model.generate()
with torch.no_grad():
    out_clean = m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids.shape[1]:], skip_special_tokens=True)
print(f'Clean (generate):  {repr(clean_text)}')

# KV reuse: prefill, crop, then use model.generate() with past_key_values
with torch.no_grad():
    out = m(ids, use_cache=True)
cache = out.past_key_values

total_pos = ids.shape[1]
# Crop to total_pos - 1 (KV up to second-to-last token)
cache.crop(total_pos - 1)
last_tok = ids[:, -1:]

with torch.no_grad():
    out_reuse = m.generate(
        input_ids=last_tok,
        past_key_values=cache,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
reuse_text = tok.decode(out_reuse[0][total_pos:], skip_special_tokens=True)
print(f'Reuse (generate):  {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
