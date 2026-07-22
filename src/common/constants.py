"""共通定数および学習率/ハイパーパラメータの安全制限モジュール"""

from src.common.logger import logger

# Learning Rate 安全上限値 (Step Law & Muon/AdamW 安定性基準)
# Muon(2D): Newton-Schulz直交化に伴う高勾配更新に対応するため上限を 0.0025 に制限
# AdamW(1D): 1次元パラメータ(Embedding/Norm/Bias)用の安全上限 0.0010
MAX_LR_2D = 0.0025
MAX_LR_1D = 0.0010


def clip_learning_rates(lr_2d: float, lr_1d: float, source: str = "") -> tuple[float, float]:
    """
    学習率を安全上限値にクリッピングし、値が変更された場合は警告ログを出力する。

    Args:
        lr_2d (float): Muon (2D) 用の最大学習率
        lr_1d (float): AdamW (1D) 用の最大学習率
        source (str): 呼び出し元の情報 (ログ識別用)

    Returns:
        tuple[float, float]: (クリッピング後の lr_2d, クリッピング後の lr_1d)
    """
    prefix = f"[{source}] " if source else ""
    clipped_2d = min(lr_2d, MAX_LR_2D)
    clipped_1d = min(lr_1d, MAX_LR_1D)

    if clipped_2d < lr_2d:
        logger.warning(
            f"{prefix}max_lr_2d ({lr_2d:.6f}) exceeded safety limit ({MAX_LR_2D:.6f}). "
            f"Clipped to {clipped_2d:.6f}."
        )

    if clipped_1d < lr_1d:
        logger.warning(
            f"{prefix}max_lr_1d ({lr_1d:.6f}) exceeded safety limit ({MAX_LR_1D:.6f}). "
            f"Clipped to {clipped_1d:.6f}."
        )

    return clipped_2d, clipped_1d
