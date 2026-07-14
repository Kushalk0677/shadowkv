"""Trace Qwen's numerical drift through attention layers at step 7."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Method A: model.generate() — capture hidden states at each step
with torch.no_grad():
    out_gen = m.generate(ids, max_new_tokens=7, pad_token_id=tok.pad_token_id,
                         output_hidden_states=True, return_dict_in_generate=True)
gen_hidden = out_gen.hidden_states  # list, one per generated step
# Each element is a tuple of (layers + 1, batch, seq, hidden), one per decoder layer + embedding

# Method B: manual loop with crop — capture hidden states
with torch.no_grad():
    out = m(ids, use_cache=True)
cache = out.past_key_values
cache.crop(ids.shape[1] - 1)
inp = ids[:, -1:]
kv = cache

manual_hidden = []
for step in range(7):
    with torch.no_grad():
        outs = m(input_ids=inp, past_key_values=kv, use_cache=True, output_hidden_states=True)
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    inp = nid
    kv = outs.past_key_values
    manual_hidden.append(outs.hidden_states)

# Compare hidden states at step 6 (the last step before divergence jumps)
print("Hidden state differences at each layer (step 6):")
print(f"{'Layer':>8} {'Diff':>12} {'Gen norm':>12} {'Man norm':>12}")
for layer in range(min(len(gen_hidden[6]), len(manual_hidden[6]))):
    # gen_hidden is tuple of (layers+1, batch, seq=1, hidden)
    # Each is (embedding, layer0, layer1, ..., layerN)
    gen_h = gen_hidden[6][layer]
    man_h = manual_hidden[6][layer]
    diff = (gen_h - man_h).abs().max().item()
    gnorm = gen_h.norm().item()
    mnorm = man_h.norm().item()
    marker = " <---" if diff > 1e-4 else ""
    print(f"{layer:>8} {diff:>12.2e} {gnorm:>12.4f} {mnorm:>12.4f}{marker}")

# Also check step 5 vs step 6
print("\nStep 5 → Step 6 diff change per layer:")
for layer in range(min(len(gen_hidden[5]), len(manual_hidden[5]))):
    diff5 = (gen_hidden[5][layer] - manual_hidden[5][layer]).abs().max().item()
    diff6 = (gen_hidden[6][layer] - manual_hidden[6][layer]).abs().max().item()
    growth = diff6 / max(diff5, 1e-10)
    if growth > 10:
        print(f"  Layer {layer}: diff5={diff5:.2e} → diff6={diff6:.2e} (x{growth:.0f})")
