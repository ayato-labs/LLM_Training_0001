import json
import sys
from pathlib import Path
from transformers import PreTrainedTokenizerFast

# プロジェクトルートパスの設定
sys.path.append(str(Path(__file__).resolve().parent))
import training_config as config
from src.preprocessing.exporter import export_db_to_jsonl

def analyze():
    # 1. データベースから JSONL ファイルを生成 (前処理の実行)
    print("Step 1: Running database preprocessing (exporting db to JSONL)...")
    export_db_to_jsonl()
    
    # 2. 生成された JSONL ファイルの確認
    dataset_path = Path("data/dataset.jsonl")
    if not dataset_path.exists():
        print(f"ERROR: Preprocessed dataset not found at {dataset_path}")
        return
        
    # 3. トークナイザーのロード
    print("\nStep 2: Loading tokenizer...")
    tokenizer_path = Path("data/tokenizer.json")
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    
    # 4. JSONL ファイルの読み込みと正確なトークン数のカウント
    print("\nStep 3: Analyzing dataset characters and exact tokens...")
    total_chars = 0
    total_tokens = 0
    record_count = 0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
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
    
    # 350M モデルの計算
    target_params = 350_000_000
    print(f"\nTarget Model Size:   {target_params:,} parameters (350M)")
    
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

if __name__ == "__main__":
    analyze()
