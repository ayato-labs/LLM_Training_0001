import math


class WSDScheduler:
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
        self._step += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs, strict=False):
            pg["lr"] = lr
        return lrs

    def _get_lrs(self):
        s = self._step
        result = []
        for peak_lr, stable_lr in zip(self.peak_lrs, self.stable_lrs, strict=False):
            if s <= self.warmup_steps:
                t = s / max(1, self.warmup_steps)
                lr = self.min_lr + (peak_lr - self.min_lr) * t
            elif s <= self.decay_start:
                lr = stable_lr
            else:
                t = (s - self.decay_start) / self.decay_steps
                decay = 1.0 - math.sqrt(t)
                lr = self.min_lr + (stable_lr - self.min_lr) * decay
            result.append(lr)
        return result
