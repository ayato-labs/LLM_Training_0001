"""共通定数および学習率/ハイパーパラメータの安全制限モジュール"""

from src.common.logger import logger

# Learning Rate 安全上限値 (Step Law & Muon/AdamW 安定性基準)
# Muon(2D): Keller Jordan実装/NorMuonの知見を踏まえ上限を 0.02 に緩和
# (二重ガード設計 / Single Source of Truth: ConfigおよびStepLaw双方から本モジュールを参照)
# AdamW(1D): 1次元パラメータ(Embedding/Norm/Bias)用の安全上限 0.0010
MAX_LR_2D = 0.02
MAX_LR_1D = 0.0010


def clip_learning_rates(lr_2d: float, lr_1d: float, source: str = "") -> tuple[float, float]:
    """
    学習率が安全推奨値を超えている場合に警告を出力する。
    Chinchilla法則およびHPOの計算結果を優先するため、強制カットは行わず数値を保持する。

    Args:
        lr_2d (float): Muon (2D) 用の最大学習率
        lr_1d (float): AdamW (1D) 用の最大学習率
        source (str): 呼び出し元の情報 (ログ識別用)

    Returns:
        tuple[float, float]: (lr_2d, lr_1d)
    """
    prefix = f"[{source}] " if source else ""

    if lr_2d > MAX_LR_2D:
        logger.warning(
            f"{prefix}max_lr_2d ({lr_2d:.6f}) exceeds standard limit ({MAX_LR_2D:.6f}). "
            f"Proceeding with calculated HPO/Chinchilla value."
        )

    if lr_1d > MAX_LR_1D:
        logger.warning(
            f"{prefix}max_lr_1d ({lr_1d:.6f}) exceeds standard limit ({MAX_LR_1D:.6f}). "
            f"Proceeding with calculated HPO/Chinchilla value."
        )

    return lr_2d, lr_1d
