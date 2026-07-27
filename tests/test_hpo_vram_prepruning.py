from src.common.vram_estimator import VramConfig, estimate_training_vram


def test_vram_estimator_prepruning_logic():
    """VRAM Estimator が危険な設定（90%超）を検出して枝刈り判定を出せることを検証"""
    # 物理 VRAM が 4.0GB の場合
    vram_gb = 4.0

    # 1. 安全な設定（小モデル、小バッチ）
    safe_config = VramConfig(
        n_params=37_000_000,
        hidden_size=512,
        num_layers=8,
        micro_batch_size=2,
        seq_len=512,
        total_vram_gb=vram_gb,
    )
    est_safe = estimate_training_vram(safe_config)
    assert est_safe.breakdown.is_safe is True
    assert est_safe.breakdown.total_estimated_gb <= (vram_gb * 0.90)

    # 2. 危険な設定（チェックポイントなし + 巨大マイクロバッチで VRAM 90% 閾値超え）
    danger_config = VramConfig(
        n_params=150_000_000,
        hidden_size=1024,
        num_layers=16,
        micro_batch_size=128,  # 極端に大きい
        seq_len=2048,
        checkpointing="none",  # アクティベーション非保存
        total_vram_gb=vram_gb,
    )
    est_danger = estimate_training_vram(danger_config)
    assert (not est_danger.breakdown.is_safe) or (
        est_danger.breakdown.total_estimated_gb > vram_gb * 0.90
    )
