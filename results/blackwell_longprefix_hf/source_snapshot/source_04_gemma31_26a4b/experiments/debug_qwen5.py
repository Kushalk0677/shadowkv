"""Match generation config between manual loop and model.generate()."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Check generation config
gc = m.generation_config
print(f'Generation config:')
print(f'  do_sample: {gc.do_sample}')
print(f'  temperature: {gc.temperature}')
print(f'  top_p: {gc.top_p}')
print(f'  num_beams: {gc.num_beams}')

# Clean with matching config
with torch.no_grad():
    out_gen = m.generate(
        ids, 
        max_new_tokens=16,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
gen_text = tok.decode(out_gen[0][ids.shape[1]:], skip_special_tokens=True)
print(f'\nmodel.generate(): {repr(gen_text)}')

# Manual loop matching the same logic
with torch.no_grad():
    out = m(ids, use_cache=True)
total_pos = ids.shape[1]
cache = out.past_key_values

gen_ids = []
last = None
for step in range(16):
    if step == 0:
        inp = ids
        kv = None
    else:
        inp = last
        kv = cache
    with torch.no_grad():
        outs = m(input_ids=inp, past_key_values=kv, use_cache=True)
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    last = nid
    cache = outs.past_key_values
    gen_ids.append(nid.item())
    if nid.item() == tok.eos_token_id:
        break

manual_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'Manual loop:      {repr(manual_text)}')
print(f'Match: {gen_text == manual_text}')
