import tempfile
from pathlib import Path

import yaml

from src.scaling_laws.calculator import calculate_chinchilla_scaling
from src.training.config import _normalize_config


def test_chinchilla_to_training_pipeline():
    """Scaling Laws 出力 → config.py → 正しい max_steps の全工程"""
    # 1. Scaling Laws 計算
    chinchilla_result = calculate_chinchilla_scaling(
        target_hours=24.0,
        user_throughput_tps=14338.6,
        user_seq_len=512,
    )

    D = chinchilla_result["computable_tokens"]

    # 2. scaling_config.yaml 相当の生成
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "scaling_config.yaml"
        yaml_content = {
            "metadata": {"computable_tokens": D},
            "training": {"seq_len": 512, "batch_size_seqs": 16},
            "model": {"target_params": chinchilla_result["recommended_architecture"]["n_params"]},
        }
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        # 3. config.py で読み込み・正規化
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        config = _normalize_config(raw)

        # 4. 検証
        expected_max_steps = D // (16 * 512)
        assert config["max_steps"] == expected_max_steps
        assert config["max_steps"] > 100_000  # 786 のバグ再発防止


def test_chinchilla_hpo_batch_size_change_updates_max_steps():
    """HPOで batch_size_seqs が変われば max_steps も再計算される"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "scaling_config.yaml"
        yaml_content = {
            "metadata": {"computable_tokens": 1_240_000_000},
            "training": {"seq_len": 512, "batch_size_seqs": 16},  # 初期値
            "model": {"target_params": 37_000_000},
        }
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        # base_config 相当
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # HPO前: batch=16
        config_before = _normalize_config(raw)
        expected_before = 1_240_000_000 // (16 * 512)
        assert config_before["max_steps"] == expected_before

        # HPO後: batch_size_seqs が 32 に変更された想定
        raw["training"]["batch_size_seqs"] = 32
        config_after = _normalize_config(raw)
        expected_after = 1_240_000_000 // (32 * 512)
        assert config_after["max_steps"] == expected_after

        # batch が2倍なら steps は半分
        assert config_after["max_steps"] == config_before["max_steps"] // 2


def test_chinchilla_computable_tokens_not_overridden_by_hpo():
    """computable_tokens は HPO で変わらない（不変）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "chinchilla_config.yaml"
        yaml_content = {
            "metadata": {"computable_tokens": 1_240_000_000},
            "training": {"seq_len": 512, "batch_size_seqs": 16},
            "model": {"target_params": 37_000_000},
        }
        with open(config_path, "w") as f:
            yaml.dump(yaml_content, f)

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        config = _normalize_config(raw)

        # computable_tokens は metadata に保持される（上書きされない）
        assert raw["metadata"]["computable_tokens"] == 1_240_000_000
