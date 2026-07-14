"""Debug: Check if token counts are aligned in prefix_shuffle."""
import json
import random
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# Load the results raw data
r = json.load(open('v10/fidelity_equiv_v7/tinyllama_results.json'))

# Reconstruct prompts from the v2 experiment
# Actually, we need the original prompts. Let me just test the shuffling logic
test_text = "Amanda: I'm so excited about the party tomorrow! Jerry: I know, I've been planning it for weeks. Neville: (smiling) I do. It was a beautiful ceremony. Luna: (beaming) It really was!"

ids_orig = tok.encode(test_text, truncation=True, max_length=384)
split = len(ids_orig) * 3 // 4
suffix = ids_orig[split:]
shuffled = suffix.copy()
random.shuffle(shuffled)
ids_mod = ids_orig[:split] + shuffled
text_mod = tok.decode(ids_mod)
ids_mod2 = tok.encode(text_mod, truncation=True, max_length=384)

print(f'Original ID count: {len(ids_orig)}')
print(f'Modified ID count: {len(ids_mod2)}')
print(f'Shared (from orig): {split}')
print(f'Shared prefix IDs match: {ids_orig[:split] == ids_mod[:split]}')

# Now check: do the first `split` tokens of the modified prompt match the originals?
if len(ids_mod2) >= split:
    print(f'First {split} mod tokens match orig: {ids_mod2[:split] == ids_orig[:split]}')
else:
    print(f'Modified text shorter than split! mod_len={len(ids_mod2)}, split={split}')
    print(f'First {len(ids_mod2)} mod tokens match orig: {ids_mod2 == ids_orig[:len(ids_mod2)]}')

# Compare token-by-token
print(f'\nOrig tokens: {ids_orig[:20]}...')
print(f'Mod tokens:  {ids_mod[:20]}...')
