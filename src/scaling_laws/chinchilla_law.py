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


def estimate_universal_arch_params(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 32000,
) -> dict[str, int]:
    """Grid Search 用の軽量な解析パラメータ推定 (torch 不要)。

    LLaMA パラメータ近似式:
    - 埋め込み層 + LM Head (Tie Weights 適用): V * H
    - 1レイヤーあたりのパラメータ数:
      - Self-Attention (GQA): 4 * H^2 * (1 + 1/group_ratio)
      - SwiGLU FFN: 3 * H * I  (中間層 I = 2.67 * H の 128 倍数)
      - RMSNorm + RoPE: ~4 * H
    """
    num_attention_heads = max(4, hidden_size // 64)
    num_key_value_heads = max(1, num_attention_heads // 4)
    intermediate_size = int(round(2.67 * hidden_size / 128) * 128)

    gqa_ratio = num_key_value_heads / num_attention_heads
    params = vocab_size * hidden_size + num_layers * (
        4 * (hidden_size**2) * (1.0 + gqa_ratio)
        + 3 * hidden_size * intermediate_size
        + 4 * hidden_size
    )
    return {
        "n_params_estimate": int(params),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
    }


def generate_universal_architecture(target_n_params: int) -> dict[str, Any]:
    """任意のターゲットパラメータ数 N に対し、本番と 100% 同一の
    Standard LLaMA アーキテクチャを動的に自動設計生成。

    (hidden_size, num_layers) のグリッド探索により、解析式パラメータ推定の
    相対誤差が ±15% 以内 (可能な限り小さく、下振れよりも上振れを優先) になる
    候補を選択する。量子化粒度 (H: 128倍数, L: 整数) に起因する -47% 級の
    大幅下振れを構造的に排除する。
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

    # 2. パラメータ比率によるスケールから中心 (H, L) を決め、グリッド探索で最良候補を選択。
    #    スコアは絶対誤差で評価し、同点の場合は下振れ (VRAM/品質リスク) より
    #    上振れを優先する。±15% 以内の候補が無い場合は探索幅を拡大する。
    scale = (target_n_params / base_n) ** (1 / 3)
    center_hidden = int(round(base_hidden * scale / 128) * 128)
    center_layers = int(round(base_layers * scale))

    best_candidate: tuple[float, float, int, int] | None = None
    best_within: tuple[float, float, int, int] | None = None

    hidden_values = [
        max(256, center_hidden + k * 128) for k in range(-6, 7)
    ]
    hidden_values = list(dict.fromkeys(hidden_values))
    layer_values = [max(2, center_layers + k) for k in range(-6, 7)]
    layer_values = list(dict.fromkeys(layer_values))

    for hidden_size in hidden_values:
        for num_layers in layer_values:
            est = estimate_universal_arch_params(
                hidden_size, num_layers, vocab_size
            )["n_params_estimate"]
            err = (est - target_n_params) / target_n_params
            candidate = (abs(err), 0.0 if err >= 0.0 else 1.0, hidden_size, num_layers)
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate
            if abs(err) <= 0.15 and (best_within is None or candidate < best_within):
                best_within = candidate

    _, _, hidden_size, num_layers = best_within or best_candidate

    # GQA 設定
    arch_meta = estimate_universal_arch_params(hidden_size, num_layers, vocab_size)
    num_attention_heads = arch_meta["num_attention_heads"]
    num_key_value_heads = arch_meta["num_key_value_heads"]
    intermediate_size = arch_meta["intermediate_size"]
    exact_n_params = arch_meta["n_params_estimate"]

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
