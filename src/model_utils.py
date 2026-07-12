"""Model Utilities: モデル初期化・パラメータ推定"""

from transformers import LlamaConfig


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
