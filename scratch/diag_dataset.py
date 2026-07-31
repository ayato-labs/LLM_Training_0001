import json
import sys

print("Diagnosing dataset.jsonl...")
with open("data/dataset.jsonl", "r", encoding="utf-8", errors="ignore") as f:
    for i in range(10):
        line = f.readline()
        if not line:
            break
        try:
            d = json.loads(line)
            keys = list(d.keys())
            text_len = len(d.get("text", ""))
            print(f"Line {i}: keys={keys}, text_len={text_len}")
        except Exception as e:
            print(f"Line {i}: JSON parse error: {e}")
