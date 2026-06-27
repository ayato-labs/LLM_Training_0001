from pathlib import Path

# --- ハードウェア環境 ---
VRAM_LIMIT_GB = 4.0
PRECISION_BYTES = 2   # 16bit

# --- 理論・学習ロジックの係数 ---
# データ対パラメータ比（スケーリング則の係数）
CHINCHILLA_RATIO = 100
# メモリオーバーヘッド係数（勾配チェックポインティング + オプティマイザ状態）
MEMORY_OVERHEAD = 1.5

# --- パス ---
ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data/dataset.jsonl"
OUTPUT_DIR = ROOT_DIR / "models/output"
TOKENIZER_PATH = ROOT_DIR / "data/tokenizer.json"