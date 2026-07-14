"""Debug sample filtering for low-count datasets."""
import warnings, sys
warnings.filterwarnings('ignore')
from datasets import load_dataset

# banking77
try:
    ds = load_dataset('mteb/banking77', split='train', trust_remote_code=True)
    count = 0
    for i in range(150):
        text = str(ds[i].get('text', ''))
        if len(text) > 20:
            count += 1
    sys.stdout.write(f'banking77: {count}/150 pass >20 char filter (total rows: {len(ds)})\n')
    if count < 10:
        # Show first few texts
        for i in range(5):
            t = str(ds[i].get('text', ''))
            sys.stdout.write(f'  [{i}] len={len(t)}: {repr(t[:60])}\n')
except Exception as e:
    sys.stdout.write(f'banking77 error: {e}\n')

# alpaca_eval
try:
    ds = load_dataset('Thanmay/alpaca_eval', split='eval', trust_remote_code=True)
    count = 0
    for i in range(150):
        text = str(ds[i].get('instruction', ''))
        if len(text) > 20:
            count += 1
    sys.stdout.write(f'alpaca_eval: {count}/150 pass >20 char filter (total: {len(ds)})\n')
    if count < 10:
        for i in range(5):
            t = str(ds[i].get('instruction', ''))
            sys.stdout.write(f'  [{i}] len={len(t)}: {repr(t[:60])}\n')
except Exception as e:
    sys.stdout.write(f'alpaca_eval error: {e}\n')

# daily_dialog
try:
    ds = load_dataset('DeepPavlov/daily_dialog', split='train', trust_remote_code=True)
    count = 0
    for i in range(150):
        dialog = ds[i].get('dialog', [])
        text = ' '.join(dialog) if isinstance(dialog, list) else str(dialog)
        if len(text) > 20:
            count += 1
    sys.stdout.write(f'daily_dialog: {count}/150 pass >20 char filter (total: {len(ds)})\n')
    if count < 10:
        for i in range(5):
            d = ds[i].get('dialog', [])
            t = ' '.join(d) if isinstance(d, list) else str(d)
            sys.stdout.write(f'  [{i}] len={len(t)}: {repr(t[:60])}\n')
except Exception as e:
    sys.stdout.write(f'daily_dialog error: {e}\n')
