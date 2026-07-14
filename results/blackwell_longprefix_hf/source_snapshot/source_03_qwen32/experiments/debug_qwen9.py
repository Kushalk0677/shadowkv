"""Compare logits at first generated step: prefill vs incremental."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])
n = ids.shape[1]
print(f'Prompt length: {n}')

# Method 1: Full prefill (process all tokens at once)
with torch.no_grad():
    out_full = m(ids, use_cache=True)
logits_full = out_full.logits[0, -1]  # logits at last position
token_full = logits_full.argmax(-1).item()
print(f'Full prefill: token={token_full} ({repr(tok.decode([token_full]))})')

# Method 2: Prefill n-1, then last token with cache
ids_prefix = ids[:, :-1]
ids_last = ids[:, -1:]
with torch.no_grad():
    out_prefix = m(ids_prefix, use_cache=True)
    out_last = m(ids_last, past_key_values=out_prefix.past_key_values, use_cache=True)
logits_last = out_last.logits[0, -1]
token_last = logits_last.argmax(-1).item()
print(f'Prefill n-1 + last tok: token={token_last} ({repr(tok.decode([token_last]))})')

# Compare logits
diff = (logits_full - logits_last).abs().max().item()
print(f'Max logit diff: {diff:.2e}')
print(f'Argmax match: {token_full == token_last}')
print(f'Top-5 full: {logits_full.topk(5).indices.tolist()}')
print(f'Top-5 last:  {logits_last.topk(5).indices.tolist()}')
