"""Match first generated token from both methods."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Force pure greedy: set generation config
m.generation_config.do_sample = False
m.generation_config.temperature = None
m.generation_config.top_p = None

# Run generate
with torch.no_grad():
    out_gen = m.generate(ids, max_new_tokens=16, pad_token_id=tok.pad_token_id)
gen_text = tok.decode(out_gen[0][ids.shape[1]:], skip_special_tokens=True)
print(f'model.generate(): {repr(gen_text)}')

# Manual loop that mimics generate exactly
with torch.no_grad():
    out = m(ids, use_cache=True)

gen_ids = []
past = None
next_input = ids
for step in range(16):
    with torch.no_grad():
        outs = m(input_ids=next_input, past_key_values=past, use_cache=True)
    next_id = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    next_input = next_id
    past = outs.past_key_values
    gen_ids.append(next_id.item())
    if next_id.item() == tok.eos_token_id:
        break

manual_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'Manual loop:      {repr(manual_text)}')
print(f'Match: {gen_text == manual_text}')

# Show first generated token
print(f'\nFirst token:')
print(f'  generate: {repr(tok.decode([out_gen[0][ids.shape[1]]], skip_special_tokens=True))}')
print(f'  manual:   {repr(tok.decode([gen_ids[0]], skip_special_tokens=True))}')
print(f'  IDs match: {out_gen[0][ids.shape[1]] == gen_ids[0]}')
