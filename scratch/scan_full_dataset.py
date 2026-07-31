import json
import time

print("Scanning full dataset.jsonl for exact text chars...")
t0 = time.time()
total_chars = 0
line_count = 0
error_count = 0

with open("data/dataset.jsonl", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line_count += 1
        try:
            d = json.loads(line)
            total_chars += len(d.get("text", ""))
        except Exception:
            error_count += 1

elapsed = time.time() - t0
print(f"Scanned {line_count:,} lines in {elapsed:.2f}s (Errors: {error_count})")
print(f"Total 'text' chars: {total_chars:,}")
print(f"Estimated tokens (@0.6 token/char for Japanese): {total_chars * 0.6:,.0f}")
print(f"Estimated tokens (@0.8 token/char for Subword/Spiece): {total_chars * 0.8:,.0f}")
