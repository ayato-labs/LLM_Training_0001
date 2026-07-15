import math


class WSDScheduler:
    """Warmup-Stable-Decay学習率スケジューラー。

    3段階で学習率を制御：
    1. Warmup: 初期学習率からピーク学習率へ線形増加
    2. Stable: ピーク学習率の一定比率（stable_ratio）を維持
    3. Decay: 最終学習率まで平方根ベースの減衰

    Args:
        optimizer: オプティマイザ
        peak_lr_2d: 2Dパラメータ（Muon等）のピーク学習率
        peak_lr_1d: 1Dパラメータ（AdamW等）のピーク学習率
        total_steps: 総ステップ数
        warmup_frac: ウォームアップ期間の割合（デフォルト0.01=1%）
        decay_frac: 減衰期間の割合（デフォルト0.20=20%）
        min_lr: 最小学習率（デフォルト1e-5）
        stable_ratio: 安定期間での学習率比率（デフォルト0.55）
    """

    def __init__(
        self,
        optimizer,
        peak_lr_2d,
        peak_lr_1d,
        total_steps,
        warmup_frac=0.01,
        decay_frac=0.20,
        min_lr=1e-5,
        stable_ratio=0.55,
    ):
        self.optimizer = optimizer
        self.peak_lrs = [peak_lr_2d, peak_lr_1d]
        self.stable_lrs = [lr * stable_ratio for lr in self.peak_lrs]
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)
        self.decay_start = int(total_steps * (1 - decay_frac))
        self.decay_steps = total_steps - self.decay_start
        self._step = 0

    def step(self):
        """1ステップ進める。"""
        self._step += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs, strict=False):
            pg["lr"] = lr
        return lrs

    def _get_lrs(self):
        """現在のステップに対する学習率を計算。"""
        s = self._step
        result = []
        for peak_lr, stable_lr in zip(self.peak_lrs, self.stable_lrs, strict=False):
            if s <= self.warmup_steps:
                # ウォームアップ期間：線形増加
                t = s / max(1, self.warmup_steps)
                lr = self.min_lr + (peak_lr - self.min_lr) * t
            elif s <= self.decay_start:
                # 安定期間：一定比率
                lr = stable_lr
            else:
                # 減衰期間：平方根減衰
                t = (s - self.decay_start) / self.decay_steps
                decay = 1.0 - math.sqrt(t)
                lr = self.min_lr + (stable_lr - self.min_lr) * decay
            result.append(lr)
        return result
