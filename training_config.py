import os
from pathlib import Path

# =====================================================================
# 1. ハードウェア・実行環境制約
# =====================================================================
# VRAM制限 (GB)。環境変数から上書き可能。未指定時はGPUの物理メモリを自動検出します。
def _detect_vram() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device_id).total_memory
            return total_mem / (1024**3)
    except Exception:
        pass
    return 32.0

VRAM_LIMIT_GB = float(os.getenv("VRAM_LIMIT_GB", _detect_vram()))
PRECISION_BYTES = 2   # 16-bit (fp16/bf16) = 2, 32-bit (fp32) = 4

# =====================================================================
# 2. スケーリング・メモリ最適化係数
# =====================================================================
CHINCHILLA_RATIO = 100  # データ対パラメータ比。過学習を防止するためのスケーリング則の基準値
MEMORY_OVERHEAD = 1.5   # メモリオーバーヘッド係数（勾配チェックポインティング + オプティマイザ状態等）

# =====================================================================
# 3. 学習のドメイン・要件設定 (人間が変更するコアパラメータ)
# =====================================================================
SEQ_LEN = 1024          # コンテキストウィンドウ長。VRAM 4GB 制約下の最大安全限界である 1024 を設定。
MAX_STEPS = 50          # 学習ステップ数 (-1でエポックベースの学習を行います)
TARGET_TOKENS = 30000000 # 学習の最適化計算で使用するターゲットの想定総トークン数（実際のデータセットと同等）
TARGET_PARAMS = 120000000 # ターゲットとするモデルのパラメータ数 (120M)
PROXY_MIN_PARAMS = 30000000 # 探索用代理モデルの最小パラメータ数下限 (30M)

# =====================================================================
# 4. ディレクトリ・パス設定
# =====================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "dataset.jsonl"
TOKENIZER_PATH = ROOT_DIR / "data" / "tokenizer.json"
OUTPUT_DIR = ROOT_DIR / "models" / "output"
