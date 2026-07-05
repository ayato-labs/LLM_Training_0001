import json
import sys
import subprocess
from pathlib import Path
from transformers import PreTrainedTokenizerFast

# プロジェクトルートパスの設定
sys.path.append(str(Path(__file__).resolve().parent.parent))
import training_config as config

# DataPreprocessing のパス
DATAPREPROCESSING_DIR = Path(__file__).resolve().parent.parent.parent / "DataPreprocessing"
DATAPREPROCESSING_VENV_PYTHON = DATAPREPROCESSING_DIR / ".venv" / "Scripts" / "python.exe"
DATASET_PATH = DATAPREPROCESSING_DIR / "data" / "dataset.jsonl"

SKIP_PIPELINE = "--skip-pipeline" in sys.argv

def analyze():
    # 1. DataPreprocessing パイプラインを実行（スキップ可能）
    if SKIP_PIPELINE and DATASET_PATH.exists():
        print("Step 1: Skipping pipeline (--skip-pipeline)...")
        print(f"  Using existing dataset: {DATASET_PATH}")
    else:
        print("Step 1: Running DataPreprocessing pipeline...")
        if not DATAPREPROCESSING_VENV_PYTHON.exists():
            print(f"ERROR: DataPreprocessing venv python not found at {DATAPREPROCESSING_VENV_PYTHON}")
            return

        try:
            result = subprocess.run(
                [
                    str(DATAPREPROCESSING_VENV_PYTHON),
                    "-m", "src.cli", "pipeline",
                    "--db", str(Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Novel_Data_Collection\novels.db")),
                    "--output", str(DATASET_PATH),
                ],
                cwd=str(DATAPREPROCESSING_DIR),
                capture_output=True,
                text=True,
                timeout=600,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"ERROR: DataPreprocessing pipeline failed:\n{result.stderr}", file=sys.stderr)
                return
        except subprocess.TimeoutExpired:
            print("ERROR: DataPreprocessing pipeline timed out (10 minutes)", file=sys.stderr)
            return
    
    # 2. 生成された JSONL ファイルの確認
    if not DATASET_PATH.exists():
        print(f"ERROR: Preprocessed dataset not found at {DATASET_PATH}")
        return
        
    # 3. トークナイザーのロード
    print("\nStep 2: Loading tokenizer...")
    tokenizer_path = Path(__file__).resolve().parent.parent / "data" / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    
    # 4. JSONL ファイルの読み込みと正確なトークン数のカウント
    print("\nStep 3: Analyzing dataset characters and exact tokens...")
    total_chars = 0
    total_tokens = 0
    record_count = 0
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text:
                    continue
                
                # 文字数
                total_chars += len(text)
                
                # トークン数 (FastTokenizerを使用して高速化)
                tokens = tokenizer.encode(text)
                total_tokens += len(tokens)
                
                record_count += 1
                if record_count % 1000 == 0:
                    print(f"  Processed {record_count} chapters...")
            except Exception as e:
                print(f"  Error processing line: {e}")
                
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Total Chapters:      {record_count:,}")
    print(f"Total Characters:    {total_chars:,}")
    print(f"Total Exact Tokens:  {total_tokens:,}")
    print(f"Avg Tokens/Chapter:  {total_tokens / record_count:.1f}")
    print(f"Avg Chars/Chapter:   {total_chars / record_count:.1f}")
    print(f"Token/Char Ratio:    {total_tokens / total_chars:.4f}")
    print("="*50)
    
    # ターゲットモデルの計算 (config から動的読み込み)
    target_params = getattr(config, "TARGET_PARAMS", 120_000_000)
    print(f"\nTarget Model Size:   {target_params:,} parameters ({target_params / 1_000_000:.0f}M)")
    
    # --- Chinchilla 20x Ratio ---
    req_tokens_20 = target_params * 20
    diff_tokens_20 = max(0, req_tokens_20 - total_tokens)
    ratio_20 = req_tokens_20 / total_tokens
    
    print("\n[Case 1: Chinchilla Compute-Optimal (Ratio = 20)]")
    print(f"  Required Tokens:   {req_tokens_20:,}")
    print(f"  Deficient Tokens:  {diff_tokens_20:,}")
    print(f"  Multiplier Needed: {ratio_20:.2f}x (約 {ratio_20:.1f} 倍のデータ)")
    print(f"  Estimated Chapters Needed: {diff_tokens_20 / (total_tokens / record_count):,.0f} chapters")
    
    # --- Current Config 100x Ratio ---
    req_tokens_100 = target_params * 100
    diff_tokens_100 = max(0, req_tokens_100 - total_tokens)
    ratio_100 = req_tokens_100 / total_tokens
    
    print("\n[Case 2: Current Pipeline Config (Ratio = 100)]")
    print(f"  Required Tokens:   {req_tokens_100:,}")
    print(f"  Deficient Tokens:  {diff_tokens_100:,}")
    print(f"  Multiplier Needed: {ratio_100:.2f}x (約 {ratio_100:.1f} 倍のデータ)")
    print(f"  Estimated Chapters Needed: {diff_tokens_100 / (total_tokens / record_count):,.0f} chapters")
    print("="*50)

    # --- 日本語のMarkdown結果ファイルを生成 ---
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    result_filename = f"result_{timestamp}.md"
    result_path = Path(__file__).resolve().parent / result_filename
    
    avg_tokens = total_tokens / record_count
    avg_chars = total_chars / record_count
    token_char_ratio = total_tokens / total_chars
    
    markdown_content = f"""# データセット トークン分析結果 ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})

## 1. 基本統計データ

| 項目 | 分析値 |
| :--- | :--- |
| **総チャプター数** | {record_count:,} チャプター |
| **総文字数** | {total_chars:,} 文字 |
| **総トークン数 (正確値)** | {total_tokens:,} トークン |
| **1チャプターあたりの平均トークン数** | {avg_tokens:.1f} トークン |
| **1チャプターあたりの平均文字数** | {avg_chars:.1f} 文字 |
| **トークン / 文字 比率** | {token_char_ratio:.4f} |

---

## 2. {target_params / 1_000_000:.0f}M モデル（ターゲット規模）の必要トークン数シミュレーション

ターゲットパラメータ数: **{target_params:,} パラメータ ({target_params / 1_000_000:.0f}M)**

### 【ケース 1】 Chinchilla 最適化構成 (Ratio = 20)
*計算量的に最も効率的なデータ対パラメータ比（20倍）を適用した場合*

- **必要トークン数**: {req_tokens_20:,} トークン
- **不足トークン数**: {diff_tokens_20:,} トークン
- **必要データ倍率**: {ratio_20:.2f}倍 (約 {ratio_20:.1f} 倍のデータ規模が必要)
- **不足分を補うために必要な推測チャプター数**: {diff_tokens_20 / avg_tokens:,.0f} チャプター

### 【ケース 2】 現行パイプライン構成 (Ratio = 100)
*過学習を防ぎつつ十分な汎化性能を獲得するために一般的に推奨されるデータ対パラメータ比（100倍）を適用した場合*

- **必要トークン数**: {req_tokens_100:,} トークン
- **不足トークン数**: {diff_tokens_100:,} トークン
- **必要データ倍率**: {ratio_100:.2f}倍 (約 {ratio_100:.1f} 倍のデータ規模が必要)
- **不足分を補うために必要な推測チャプター数**: {diff_tokens_100 / avg_tokens:,.0f} チャプター

---
*※ 本レポートは `analyze_dataset_tokens.py` の実行により自動生成されました。*
"""
    
    try:
        with open(result_path, "w", encoding="utf-8") as rf:
            rf.write(markdown_content)
        print(f"\n[Success] 日本語の分析レポートを以下に作成しました:\n  {result_path.resolve()}", flush=True)
    except Exception as e:
        print(f"\n[Error] レポートファイルの書き込みに失敗しました: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    analyze()
