"""Muon Optimizer: 2D パラメータ（重み行列）専用最適化

Reference: https://github.com/KellerJordan/Muon
2D以外のパラメータ（embedding, bias, LayerNorm）は AdamW で別途最適化。
"""

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Muon optimizer for 2D parameters (weight matrices).

    Args:
        params: 2Dパラメータのみ（ndim==2）
        lr: 学習率（推奨: max_lr_2d と同一）
        momentum: モメンタム係数（デフォルト: 0.95）
        nesterov: ネステロフモメンタム（デフォルト: True）
    """

    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue  # 1Dはスキップ（AdamW担当）

                grad = p.grad
                state = self.state[p]

                # モメンタムバッファ初期化
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)

                # 更新方向計算（ネステロフ対応）
                if group["nesterov"]:
                    update = grad.add(buf, alpha=group["momentum"])
                else:
                    update = buf

                # Newton-Schulz直交化（最新論文推奨の3反復で高速化）
                update = self._newton_schulz(update, steps=3)

                # パラメータ更新
                p.add_(update, alpha=-group["lr"])

    def _newton_schulz(self, G: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """Newton-Schulz iteration for orthogonalization.

        5x5 Toeplitz coefficients (最適化済み):
        a=3.4445, b=-4.7750, c=2.03153
        """
        a, b, c = 3.4445, -4.7750, 2.03153
        # 数数値的オーバーフロー（fp16溢れ）および精度低下を防ぐため、
        # 計算過程のみ float32 にキャストする
        G_fp32 = G.to(torch.float32)
        X = G_fp32 / (G_fp32.norm() + 1e-7)

        # Newton-Schulz requires rows <= cols. If rows > cols, operate on the transpose.
        transposed = G.size(0) > G.size(1)
        if transposed:
            X = X.T

        for _range in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * (A @ B)

        if transposed:
            X = X.T

        return X.to(G.dtype)
