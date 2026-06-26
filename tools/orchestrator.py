import json
import subprocess
import sys
from pathlib import Path
from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target

def scale_hpo_to_target(proxy_hpo, target_params):
    """
    Step Law の数式に基づき、ProxyモデルのパラメータをTargetモデルへ外挿する。
    本システムは compute_hpo_for_target ですでにn_paramsを引数にとっているため、
    実質的には再度 Target params で計算することで外挿と同義となる。
    """
    return compute_hpo_for_target(
        n_params=target_params, 
        n_tokens=proxy_hpo['n_tokens'], 
        seq_len=proxy_hpo['seq_len']
    )

def orchestrate():
    # 1. ハードウェア制約 (RTX 3050 4GB VRAM)
    target_params = 125_000_000 # 限界値
    proxy_params = 70_000_000   # 探索用モデル
    n_tokens = 5_000_000
    
    print(f"Orchestrator: Starting Proxy Exploration (Size: {proxy_params})")
    # 2. 探索フェーズ
    # hpo_manager を将来的に統合予定。現時点では Step Law の推奨値をベースにする
    proxy_hpo = compute_hpo_for_target(n_params=proxy_params, n_tokens=n_tokens, seq_len=512)
    
    print(f"Orchestrator: Scaling to Target (Size: {target_params})")
    # 3. 外挿フェーズ
    final_hpo = scale_hpo_to_target(proxy_hpo, target_params)
    
    # 4. 学習設定の生成
    run_config = {
        "model_params": {
            "n_params": target_params,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "hpo": final_hpo,
        "data_path": "data/dataset.jsonl"
    }
    
    # 一時的な設定ファイルを保存
    config_path = Path("current_run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
        
    print(f"Orchestrator: Run config generated: {config_path}")
    print(f"HPO Params: {final_hpo}")
    
    # 5. 学習実行
    # train_model.py が current_run_config.json を引数として受け取る
    print("Orchestrator: Launching training...")
    try:
        subprocess.run([sys.executable, "src/train_model.py", str(config_path)], check=True)
        print("Orchestrator: Training finished.")
    except subprocess.CalledProcessError as e:
        print(f"Orchestrator: Training failed: {e}")

if __name__ == "__main__":
    orchestrate()
