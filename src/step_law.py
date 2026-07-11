# utils/step_law.py
# 論文: arXiv:2503.04715, 数式は Step Law 公式サイトより


def step_law_optimal_lr(n_params: int, n_tokens: int) -> float:
    """
    Step Law による最適 Learning Rate の計算。
    正規化と上限制限を追加して小規模モデルでの発散を防止。
    """
    # パラメータ数を 1M (10^6) 単位に正規化
    n_params_m = n_params / 1e6

    # 公式: η*(N, D) = 1.79 * N^(-0.713) * D^(0.307)
    lr = 1.79 * (n_params_m**-0.713) * (n_tokens**0.307)

    # 【安全装置】学習率が発散しないよう上限(0.01)を設ける
    return min(lr, 0.01)


def step_law_optimal_batch(n_tokens: int) -> int:
    """
    Step Law による最適 Batch Size（トークン数単位）の計算。

    式: B*(D) = 0.58 * D^(0.571)

    Args:
        n_tokens: 学習に使用するトークン数

    Returns:
        最適 batch size（トークン数単位）
    """
    b_tokens = 0.58 * (n_tokens**0.571)
    return int(b_tokens)


def compute_hpo_for_target(
    n_params: int,
    n_tokens: int,
    seq_len: int = 512,
) -> dict:
    """
    ターゲットモデルの HPO を Step Law から一発計算。
    グリッドサーチ不要。

    Args:
        n_params: パラメータ数（embedding 除く）
        n_tokens: 学習トークン数
        seq_len:  シーケンス長

    Returns:
        HPO 設定の辞書
    """
    max_lr = step_law_optimal_lr(n_params, n_tokens)
    batch_tokens = step_law_optimal_batch(n_tokens)
    batch_seqs = max(1, batch_tokens // seq_len)

    # Muon LR / AdamW 1D LR の推奨比率（IMU-1 実験より）
    muon_lr = max_lr
    adamw_lr = max_lr * (0.007 / 0.0235)  # ≈ max_lr * 0.298

    return {
        "n_params": n_params,
        "n_tokens": n_tokens,
        "seq_len": seq_len,
        "max_lr_2d": round(muon_lr, 6),  # NorMuon / Muon 用
        "max_lr_1d": round(adamw_lr, 6),  # AdamW (embedding, bias, LN) 用
        "min_lr": 1e-5,  # ★ 固定（Step Law 推奨）
        "batch_size_tokens": batch_tokens,
        "batch_size_seqs": batch_seqs,
        "stable_lr_ratio": 0.55,  # IMU-1: stable LR = peak × 0.55
        "stable_lr_2d": round(muon_lr * 0.55, 6),
        "warmup_ratio": 0.03,  # Step Law / Muon recommended warmup ratio
    }
