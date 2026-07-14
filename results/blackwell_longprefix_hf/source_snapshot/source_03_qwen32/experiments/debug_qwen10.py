"""Compare step-by-step logits: model.generate() vs manual loop."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()
m.generation_config.do_sample = False

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Method A: model.generate()
with torch.no_grad():
    out_gen = m.generate(ids, max_new_tokens=16, pad_token_id=tok.pad_token_id,
                         output_logits=True, return_dict_in_generate=True)
gen_logits = out_gen.logits  # list of tensors, one per generated step

# Method B: manual loop with crop
with torch.no_grad():
    out = m(ids, use_cache=True)
cache = out.past_key_values
cache.crop(ids.shape[1] - 1)
inp = ids[:, -1:]
kv = cache

manual_logits = []
for step in range(16):
    with torch.no_grad():
        outs = m(input_ids=inp, past_key_values=kv, use_cache=True)
    logits = outs.logits[0, -1]
    manual_logits.append(logits)
    nid = logits.argmax(-1)
    inp = nid.unsqueeze(0).unsqueeze(0)
    kv = outs.past_key_values
    if nid.item() == tok.eos_token_id:
        break

# Compare logits at each step
print(f'{"Step":>5} {"Gen token":>15} {"Man token":>15} {"Max logit diff":>15} {"Argmax match":>12}')
for step in range(min(len(gen_logits), len(manual_logits))):
    gl = gen_logits[step][0] if gen_logits[step].dim() > 1 else gen_logits[step]
    ml = manual_logits[step]
    diff = (gl - ml).abs().max().item()
    gen_tok = gl.argmax(-1).item()
    man_tok = ml.argmax(-1).item()
    gen_text = tok.decode([gen_tok])
    man_text = tok.decode([man_tok])
    print(f'{step:>5} {repr(gen_text):>15} {repr(man_text):>15} {diff:>15.2e} {gen_tok==man_tok!s:>12}')
