"""Use model.generate() with full suffix approach."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Clean generation  
with torch.no_grad():
    out_clean = m.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
clean_text = tok.decode(out_clean[0][ids.shape[1]:], skip_special_tokens=True)
print(f'Clean:             {repr(clean_text)}')

# Approach: prefill + model.generate() with last token + cropped cache
with torch.no_grad():
    out = m(ids, use_cache=True)

total = ids.shape[1]
cache = out.past_key_values
cache.crop(total - 1)  # KV for positions 0..total-2
last_tok = ids[:, -1:]  # token at position total-1

with torch.no_grad():
    out_reuse = m.generate(
        input_ids=last_tok,
        past_key_values=cache,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        # Explicitly set greedy search params
        use_cache=True,
        num_beams=1,
        early_stopping=False,
    )
# The output includes: last_tok (1) + 16 generated = 17 tokens
# But we want only the 16 generated tokens
reuse_text = tok.decode(out_reuse[0][1:], skip_special_tokens=True)
print(f'Reuse (generate):  {repr(reuse_text)}')
print(f'Match: {clean_text == reuse_text}')
