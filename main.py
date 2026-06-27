import json
import subprocess
import sys
import os
from pathlib import Path
from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target
import training_config as config
from src.preprocessing.exporter import export_db_to_jsonl

def get_optimal_target_params(n_tokens):
    """
    データトークン量とVRAM制約から、理論とハードウェアの両制約を満たす最大サイズを動的に算出する
    """
    # 1. Scaling Law による推奨値 (N = D / Ratio)
    n_opt = n_tokens // config.CHINCHILLA_RATIO
    
    # 2. メモリ制約による理論上の最大値
    vram_bytes = config.VRAM_LIMIT_GB * (1024**3)
    n_max = int(vram_bytes / (config.PRECISION_BYTES * config.MEMORY_OVERHEAD))
    
    # 3. 両方の制約を満たす動的最小値
    return min(n_opt, n_max)

def run_experiment_dynamic(params, tokens, lr, steps, proxy_hidden, proxy_layers, proxy_heads, seq_len=512):
    """ 指定パラメータで短時間学習を行い、最終Lossを返す """
    hpo = compute_hpo_for_target(n_params=params, n_tokens=tokens, seq_len=seq_len)
    hpo['max_lr_2d'] = lr

    run_config = {
        "model_params": {
            "n_params": params, 
            "hidden_size": proxy_hidden, 
            "num_hidden_layers": proxy_layers, 
            "num_attention_heads": proxy_heads
        },
        "hpo": hpo,
        "data_path": str(config.DATA_PATH),
        "max_steps": steps
    }
    
    config_path = Path("experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
        
    subprocess.run([sys.executable, "src/train.py", str(config_path)], check=True)
    
    with open("last_run_result.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("train_loss", float("inf"))

def orchestrate():
    # 0. 前処理の実行 (責務：学習プロジェクトの一部)
    print("Orchestrator: Running preprocessing...")
    export_db_to_jsonl()
    
    # 1. ハードウェア制約とスケーリング定義
    n_tokens = config.TARGET_TOKENS
    target_params = get_optimal_target_params(n_tokens)
    proxy_params = int(target_params * 0.05) # 5%サイズ
    
    # ターゲットとアスペクト比を維持するための比率計算
    ratio = proxy_params / target_params
    proxy_hidden_raw = max(128, int(768 * (ratio ** 0.5)))
    proxy_layers = max(2, int(12 * ratio))
    proxy_heads = max(2, int(12 * ratio))
    
    # hidden_size が heads で割り切れ、かつ head_dim が偶数になるように調整
    head_dim = (proxy_hidden_raw // proxy_heads)
    if head_dim % 2 != 0:
        head_dim += 1
    proxy_hidden = head_dim * proxy_heads

    print(f"Orchestrator: Target {target_params} params, Proxy {proxy_params} params (H:{proxy_hidden}, L:{proxy_layers})")
    
    # 2. 探索フェーズ
    base_hpo = compute_hpo_for_target(n_params=proxy_params, n_tokens=n_tokens, seq_len=config.SEQ_LEN)
    base_lr = base_hpo['max_lr_2d']
    candidates = [base_lr * 0.5, base_lr, base_lr * 2.0]
    
    best_lr = base_lr
    min_loss = float("inf")
    
    print("Orchestrator: Starting Dynamic Proxy Exploration...")
    for lr in candidates:
        print(f"Testing LR: {lr}")
        loss = run_experiment_dynamic(proxy_params, n_tokens, lr, config.MAX_STEPS, proxy_hidden, proxy_layers, proxy_heads)
        print(f"Loss: {loss}")
        if loss < min_loss:
            min_loss = loss
            best_lr = lr
            
    print(f"Best LR found: {best_lr}")
    
    # 3. 本番用パラメータへ外挿
    final_hpo = compute_hpo_for_target(n_params=target_params, n_tokens=n_tokens, seq_len=config.SEQ_LEN)
    scaling_factor = best_lr / base_lr
    final_hpo['max_lr_2d'] *= scaling_factor
    final_hpo['max_lr_1d'] *= scaling_factor
    
    # 4. 本番実行用設定
    run_config = {
        "model_params": {
            "n_params": target_params, 
            "hidden_size": 768, 
            "num_hidden_layers": 12, 
            "num_attention_heads": 12
        },
        "hpo": final_hpo,
        "data_path": str(config.DATA_PATH)
    }
    
    with open(Path("current_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
        
    print("Orchestrator: Launching Final Training...")
    subprocess.run([sys.executable, "src/train.py", "current_run_config.json"], check=True)

if __name__ == "__main__":
    orchestrate()
