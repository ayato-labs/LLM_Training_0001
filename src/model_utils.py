"""Model Utilities: 初期化・DeepSpeed設定生成・パラメータ推定"""

from transformers import LlamaConfig
import torch
import json
import tempfile


def create_model_config(config: dict, tokenizer) -> LlamaConfig:
    """config dict から LlamaConfig 生成（hidden_size調整込み）"""
    mp = config["model_params"]
    hidden = mp["hidden_size"]
    heads = mp["num_attention_heads"]
    # hidden_size を heads の倍数に調整
    adjusted_hidden = (hidden // heads) * heads
    
    return LlamaConfig(
        vocab_size=mp["vocab_size"],
        hidden_size=adjusted_hidden,
        intermediate_size=mp["intermediate_size"],
        num_hidden_layers=mp["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=mp["num_key_value_heads"],
        rope_theta=mp["rope_theta"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,  # training用
    )


def estimate_model_size(model) -> int:
    """モデルパラメータ数推定（埋め込み込み）"""
    return sum(p.numel() for p in model.parameters())


def generate_deepspeed_config(n_params: int, vram_limit_gb: float, precision: str = "bf16") -> str:
    """DeepSpeed JSON生成→ファイル保存→パス返却"""
    
    est_vram = (n_params * 14) / (1024**3)  # BF16+Optimizer+Grad概算
    
    if precision == "bf16":
        precision_cfg = {"bf16": {"enabled": True}}
    else:
        precision_cfg = {"fp16": {"enabled": True, "loss_scale": 0}}
    
    # ZeRO Stage決定
    if est_vram > vram_limit_gb * 0.9:
        zero = {"stage": 3, "offload_param": {"device": "cpu"}, "offload_optimizer": {"device": "cpu"}}
    elif vram_limit_gb <= 5.0:
        zero = {"stage": 1}
    else:
        zero = {"stage": 2}
    
    ds_config = {
        **precision_cfg,
        "zero_optimization": zero,
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto",
        "gradient_clipping": 1.0,
    }
    
    fd, path = tempfile.mkstemp(prefix="ds_config_", suffix=".json", dir=".")
    with open(path, "w") as f:
        json.dump(ds_config, f, indent=2)
    return path
