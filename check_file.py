with open('scripts/find_hparams.py') as f:
    lines = f.readlines()
for i, l in enumerate(lines[209:225], 210):
    print(f'{i}: {repr(l)}')