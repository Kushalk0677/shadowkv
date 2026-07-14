"""Check cache length at each generation step."""
import warnings, torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cpu'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float32, low_cpu_mem_usage=True).eval()
m.generation_config.do_sample = False

prompt = "Amanda: I'm so excited about the party tomorrow!\nJerry: I know, I've been planning it for weeks."
ids = torch.tensor([tok.encode(prompt, truncation=True, max_length=384)])

# Generate and track KV lengths
print("model.generate() KV cache lengths at each step:")
with torch.no_grad():
    out_gen = m.generate(
        ids, max_new_tokens=4, pad_token_id=tok.pad_token_id,
        output_hidden_states=False, output_attentions=False,
        return_dict_in_generate=True,
    )
print(f'  Output length: {out_gen.sequences.shape[1]}')
print(f'  Generated: {repr(tok.decode(out_gen.sequences[0][ids.shape[1]:], skip_special_tokens=True))}')

# Manual loop
print("\nManual loop KV cache lengths:")
with torch.no_grad():
    out = m(ids, use_cache=True)
past = out.past_key_values
total = ids.shape[1]
print(f'  After prefill: {past.get_seq_length()} (expected {ids.shape[1]})')

gen_ids = []
next_input = ids
for step in range(4):
    with torch.no_grad():
        outs = m(input_ids=next_input, past_key_values=past, use_cache=True)
    nid = outs.logits[0, -1].argmax(-1, keepdim=True).unsqueeze(0)
    gen_ids.append(nid.item())
    past = outs.past_key_values
    if step == 0:
        # After first step, next_input should be 1 token
        pass
    next_input = nid
    print(f'  Step {step+1}: cache seq_len = {past.get_seq_length()}')

manual_text = tok.decode(gen_ids, skip_special_tokens=True)
print(f'  Generated: {repr(manual_text)}')
