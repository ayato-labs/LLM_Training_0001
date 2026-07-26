"""Post-training context window extension (RoPE / NTK-aware scaling).

Usage:
    python -m src.rope.main checkpoint=models/output/checkpoint-latest new_max=8192
    python -m src.rope.main checkpoint=... new_max=16384 method=ntk --apply
"""

import sys


def main():
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args[k.strip().lower()] = v.strip()
        elif arg.startswith("--"):
            args[arg.lstrip("-").lower()] = "true"

    checkpoint = args.get("checkpoint", "")
    new_max = int(args.get("new_max", "8192"))
    method = args.get("method", "linear")

    print(f"Extending context of {checkpoint} to {new_max} using {method}")
    print("TODO: Implement RoPE extension logic")

    if args.get("apply") == "true" or "--apply" in sys.argv:
        print("[APPLY] Not yet implemented")


if __name__ == "__main__":
    main()
