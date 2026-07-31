"""Chinchilla Scaling Law (Hoffmann et al. DeepMind 2022)

Compute-Optimalモデルパラメータ数 N および計算可能トークン数 D (D = 20N, C = 6ND)
の数学的導出および標準 LLaMA 幾何学的モデルアーキテクチャの動的生成モジュール。
"""

from typing import Any

from src.common.logger import logger


def calculate_compute_optimal_n_d(total_tokens_computable: float) -> tuple[float, float]:
    """Chinchilla Scaling Law (Hoffmann et al., 2022) に基づく最適 N, D の算出.

    Chinchilla Golden Rule: D = 20N
    Total Compute C = 6ND = 6 * N * (20N) = 120 N^2
    total_tokens_computable (D) に対し、
    Compute-Optimal Model Size N_opt = D / 20
    """
    chinchilla_n = total_tokens_computable / 20.0
    chinchilla_d = total_tokens_computable
    return chinchilla_n, chinchilla_d


def calculate_compute_optimal_n_d_from_flops(total_compute_flops: float) -> tuple[float, float]:
    """計算予算 C (FLOPs) から Compute-Optimal な N, D を導出.

    トークン/sec はモデル規模 N に依存するため、時間予算はトークンではなく
    ハードウェアの実効FLOPsレート (実測tps x プロキシ規模 x 6or8 FLOPs/token) で
    定義するのが唯一自己整合的な方法。

    Chinchilla: C = 6ND, D = 20N より C = 120 N^2 → N = sqrt(C / 120), D = 20N
    """
    chinchilla_n = (max(total_compute_flops, 0.0) / 120.0) ** 0.5
    chinchilla_d = 20.0 * chinchilla_n
    return chinchilla_n, chinchilla_d


def generate_universal_architecture(target_n_params: int) -> dict[str, Any]:
    """任意のターゲットパラメータ数 N に対し、幾何学的かつ本番と 100% 同一の
    Standard LLaMA アーキテクチャを動的に自動設計生成。

    LLaMA パラメータ近似式:
    - 埋め込み層 + LM Head (Tie Weights 適用): V * H
    - 1レイヤーあたりのパラメータ数:
      - Self-Attention (GQA): 4 * H^2 * (1 + 1/group_ratio)
      - SwiGLU FFN: 3 * H * I  (中間層 I = 2.67 * H)
      - RMSNorm + RoPE: ~4 * H
    """
    from pathlib import Path

    import torch
    import yaml

    # 1. デフォルト基準アーキテクチャ (37M 規模)
    vocab_size = 32000
    base_hidden = 512
    base_layers = 4
    base_n = 37_000_000

    config_path = Path("configs/base_config.yaml")
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "model" in cfg and "llama" in cfg["model"]:
                llama_cfg = cfg["model"]["llama"]
                vocab_size = llama_cfg.get("vocab_size", vocab_size)
        except Exception as e:
            logger.warning(
                f"Could not load base_config.yaml in generate_universal_architecture: {e}"
            )

    # パラメータ比率によるスケール
    scale = (target_n_params / base_n) ** (1 / 3)
    hidden_size = int(round(base_hidden * scale / 128) * 128)
    hidden_size = max(256, hidden_size)

    num_layers = int(round(base_layers * scale))
    num_layers = max(2, num_layers)

    # GQA 設定
    num_attention_heads = max(4, hidden_size // 64)
    num_key_value_heads = max(1, num_attention_heads // 4)

    # SwiGLU FFN 中間層次元 (128 の倍数)
    intermediate_size = int(round(2.67 * hidden_size / 128) * 128)

    embed_params = vocab_size * hidden_size
    gqa_ratio = num_key_value_heads / num_attention_heads
    attn_params = 4 * (hidden_size**2) * (1.0 + gqa_ratio)
    ffn_params = 3 * hidden_size * intermediate_size
    params_per_layer = attn_params + ffn_params
    exact_n_params = embed_params + (num_layers * params_per_layer)

    # PyTorch meta device による高精度計測検証
    try:
        from unittest.mock import MagicMock

        from transformers import LlamaForCausalLM

        from src.training.model_utils import create_model_config

        dummy_tokenizer = MagicMock()
        dummy_tokenizer.vocab_size = vocab_size

        config_dict = {
            "model": {
                "llama": {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_layers,
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "intermediate_size": intermediate_size,
                    "rope_theta": 10000.0,
                    "vocab_size": vocab_size,
                    "attn_implementation": "sdpa",
                    "tie_word_embeddings": True,
                }
            }
        }
        llama_config = create_model_config(config_dict, dummy_tokenizer)
        with torch.device("meta"):
            meta_model = LlamaForCausalLM(llama_config)
            exact_n_params = sum(p.numel() for p in meta_model.parameters())
    except Exception as e:
        logger.debug(f"Meta device count fallback: {e}")

    return {
        "n_params": int(exact_n_params),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
        "attn_implementation": "sdpa",
        "tie_word_embeddings": True,
    }


def load_base_model_defaults() -> dict[str, Any]:
    """`base_config.yaml` からベースモデルのデフォルトパラメータ数とアーキテクチャ構造を安全にロード。"""
    from pathlib import Path

    import yaml

    defaults = {
        "n_params": 86_523_648,
        "hidden_size": 768,
        "num_hidden_layers": 10,
        "num_attention_heads": 12,
        "num_key_value_heads": 3,
        "intermediate_size": 2048,
        "vocab_size": 32000,
        "rope_theta": 10000.0,
        "attn_implementation": "sdpa",
        "tie_word_embeddings": True,
    }

    config_path = Path("configs/base_config.yaml")
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "model" in cfg:
                model_cfg = cfg["model"]
                if "target_params" in model_cfg:
                    defaults["n_params"] = int(model_cfg["target_params"])
                if "llama" in model_cfg:
                    defaults.update(model_cfg["llama"])
        except Exception as e:
            logger.warning(f"Could not load base_config.yaml defaults: {e}")

    return defaults
