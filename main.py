import json
import subprocess
import sys
import os
from pathlib import Path
from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target
import training_config as config
from src.preprocessing.exporter import export_db_to_jsonl

def get_optimal_target_params():
    """
    ユーザー指定のターゲットモデルサイズを取得し、VRAM制限による理論上の限界を超えないようにキャップをかける。
    """
    target_params = getattr(config, "TARGET_PARAMS", 120_000_000)
    
    # メモリ制約による理論上の最大値
    vram_bytes = config.VRAM_LIMIT_GB * (1024**3)
    n_max = int(vram_bytes / (config.PRECISION_BYTES * config.MEMORY_OVERHEAD))
    
    return min(target_params, n_max)

def estimate_llama_dimensions(n_params):
    """
    指定パラメータ数に最も近いLlamaモデルの次元（Hidden Size, Layers, Heads）を計算する。
    Llamaのパラメータ数は大まかに: 12 * L * H^2 で近似できます。
    """
    best_L = 2
    best_H = 128
    min_diff = float('inf')
    
    for L in range(2, 26, 2):
        H_raw = int((n_params / (12 * L)) ** 0.5)
        # H は 64 の倍数に揃える
        H = max(64, (H_raw // 64) * 64)
        est = 12 * L * (H ** 2)
        diff = abs(est - n_params)
        if diff < min_diff:
            min_diff = diff
            best_L = L
            best_H = H
            
    best_heads = max(2, best_H // 64)
    # heads で割り切れるように hidden_size を調整
    best_hidden = (best_H // best_heads) * best_heads
    return best_hidden, best_L, best_heads

def run_experiment_dynamic(params, tokens, lr, steps, proxy_hidden, proxy_layers, proxy_heads, seq_len=1024):
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
    target_params = get_optimal_target_params()
    
    # 探索性能を担保するため代理モデルのサイズに下限キャップを設ける
    min_proxy = getattr(config, "PROXY_MIN_PARAMS", 30_000_000)
    proxy_params = max(int(target_params * 0.05), min_proxy)
    # 代理モデルがターゲットサイズを超えないように制限
    proxy_params = min(proxy_params, target_params)
    
    # 代理モデルの次元を動的に推定
    proxy_hidden, proxy_layers, proxy_heads = estimate_llama_dimensions(proxy_params)
    # 本番モデルの次元を動的に推定
    target_hidden, target_layers, target_heads = estimate_llama_dimensions(target_params)

    print(f"Orchestrator: Target {target_params} params (H:{target_hidden}, L:{target_layers}, Heads:{target_heads})")
    print(f"Orchestrator: Proxy {proxy_params} params (H:{proxy_hidden}, L:{proxy_layers}, Heads:{proxy_heads})")
    
    # 2. 探索フェーズ
    base_hpo = compute_hpo_for_target(n_params=proxy_params, n_tokens=n_tokens, seq_len=config.SEQ_LEN)
    base_lr = base_hpo['max_lr_2d']
    candidates = [base_lr * 0.5, base_lr, base_lr * 2.0]
    
    best_lr = base_lr
    min_loss = float("inf")
    
    print("Orchestrator: Starting Dynamic Proxy Exploration...")
    for lr in candidates:
        print(f"Testing LR: {lr}")
        loss = run_experiment_dynamic(
            proxy_params, n_tokens, lr, config.MAX_STEPS, 
            proxy_hidden, proxy_layers, proxy_heads, seq_len=config.SEQ_LEN
        )
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
            "hidden_size": target_hidden, 
            "num_hidden_layers": target_layers, 
            "num_attention_heads": target_heads
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
