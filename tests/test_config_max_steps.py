from src.training.config import _normalize_config


def test_config_calculates_max_steps_from_computable_tokens():
    """config.py が computable_tokens から max_steps を動的計算する"""
    raw = {
        "model": {"target_params": 37_000_000},
        "training": {
            "seq_len": 512,
            "batch_size_seqs": 16,
        },
        "metadata": {
            "computable_tokens": 1_240_000_000,  # Chinchilla出力
        },
    }

    config = _normalize_config(raw)

    # max_steps = 1_240_000_000 / (16 * 512) = 151,250
    expected = 1_240_000_000 // (16 * 512)
    assert config["max_steps"] == expected
    assert config["max_steps"] > 100_000  # 786 ではないこと確認


def test_config_fallback_when_no_computable_tokens():
    """computable_tokens なければ既存ロジック（num_epochs等）を使用"""
    raw = {
        "model": {"target_params": 37_000_000},
        "training": {
            "seq_len": 512,
            "batch_size_seqs": 16,
            "num_epochs": 3,
        },
        # metadata なし
    }

    config = _normalize_config(raw)
    # max_steps は計算されず、既存の -1 または num_epochs ベース
    assert "max_steps" in config


def test_config_explicit_max_steps_not_overridden():
    """明示的な max_steps がある場合は上書きされない"""
    raw = {
        "model": {"target_params": 37_000_000},
        "training": {
            "seq_len": 512,
            "batch_size_seqs": 16,
            "max_steps": 5000,  # 明示指定
        },
        "metadata": {
            "computable_tokens": 1_240_000_000,
        },
    }

    config = _normalize_config(raw)
    # 明示値が保持される
    assert config["max_steps"] == 5000


def test_config_uses_batch_size_seqs_from_raw():
    """raw.training.batch_size_seqs が優先される"""
    raw = {
        "model": {"target_params": 37_000_000},
        "training": {
            "seq_len": 512,
            "batch_size_seqs": 32,  # 異なる値
        },
        "metadata": {
            "computable_tokens": 1_240_000_000,
        },
    }

    config = _normalize_config(raw)
    # batch=32 で計算される
    expected = 1_240_000_000 // (32 * 512)
    assert config["max_steps"] == expected
