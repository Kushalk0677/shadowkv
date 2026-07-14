"""Use model.generate() with full suffix approach (not 1-token)."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()
m.generation_config.do_sample = False
m.generation_config.temperature = None
m.generation_config.top_p = None

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Clean generation
with torch.no_grad():
    out_clean = m.generate(ids, max_new_tokens=16, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids.shape[1]:], skip_special_tokens=True)
print(f'Clean (generate):  {repr(clean_text)}')

# Approach: prefill A, crop to shared, prefill B's suffix, then generate with model.generate()
# Use the full prompt B (re-encoded) as input with the cropped A cache

# Create prompt B (identical for test)
ids_b = ids.clone()

# 1. Prefill A
with torch.no_grad():
    out_a = m(ids, use_cache=True)
cache_a = out_a.past_key_values

# 2. Crop to shared (full prompt for identical test)
shared = ids.shape[1]
cache_a.crop(shared)

# 3. Prefill B's suffix on cropped cache
# Since prompts are identical, there's no suffix to prefill
# Use model.generate() with the FULL prompt B and the cropped cache (which is the full A cache)
# But this won't work because generate() will re-prefill B from scratch...

# Alternative: use model.generate() with start from the CACHED position
# Pass the cropped cache that has shared positions, and pass empty input_ids

# Actually, let's try: pass the full prompt B as input, but with the cache from A
# The model should: take B's tokens, attend to A's cached KV for overlapping tokens
# But since tokens are identical, this should give the same result

# Hmm, this won't work because the cache has KV for the OLD tokens (from A), 
# not for B's tokens.

# Simpler approach: just use the manual loop but fix the cache doubling bug
# The bug is: at step 0, we pass full prompt + None cache
# The model returns KV for the full prompt + the generated first token
# But then at step 1, we pass the generated token + the cache which includes it

# Fix: at step 0, crop the cache to total_pos - 1 before step 1
with torch.no_grad():
    out = m(ids, use_cache=True)

total_pos = ids.shape[1]
cache = out.past_key_values  # seq_len = total_pos
# At step 0: inp = ids, kv = None
# After step 0: cache has seq_len = total_pos

# For step 1: we need cache with seq_len = total_pos - 1, and inp = last token of ids
# BUT the cache at step 0 has ALL positions including the last token
# So we crop
cache.crop(total_pos - 1)
last_tok = ids[:, -1:]

gen_ids = []
inp = last_tok
kv = cache
for step in range(16):
    with torch.no_grad():
        outs = m(input_ids=inp, past_key_values=kv, use_cache=True)
        print(f'  Step {step}: cache in={kv.get_seq_length() if kv else 0}, out={outs.past_key_values.get_seq_length()}')
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    inp = nid
    kv = outs.past_key_values
    gen_ids.append(nid.item())
    if nid.item() == tok.eos_token_id:
        break

reuse_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'\nReuse (manual): {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
