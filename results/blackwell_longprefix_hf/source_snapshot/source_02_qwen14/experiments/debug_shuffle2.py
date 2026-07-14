"""Debug: verify fix using token arrays directly."""
import json
import random
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

# Use raw tokens directly
text = "Amanda: I'm so excited about the party tomorrow! Jerry: I know, I've been planning it for weeks."
ids = tok.encode(text, truncation=True, max_length=384)
print(f'Original: {len(ids)} tokens')

split = len(ids) * 3 // 4
prefix = ids[:split]
suffix = ids[split:]
shuffled = suffix.copy()
random.shuffle(shuffled)
ids_mod = prefix + shuffled

# Decode for model text input (doesn't matter for comparison)
# But for KV reuse, use `split` (computed from original) and `ids_mod` directly
print(f'Prefix tokens match: {ids[:split] == ids_mod[:split]}')
print(f'Shared count: {split}')
print(f'Original total: {len(ids)}')
print(f'Modified total: {len(ids_mod)}')

# Now verify the experiment with direct token access
print(f'\nFirst 10 orig tokens: {ids[:10]}')
print(f'First 10 mod tokens:  {ids_mod[:10]}')
print(f'Suffix orig: {ids[split:]}')
print(f'Suffix mod:  {ids_mod[split:]}')
