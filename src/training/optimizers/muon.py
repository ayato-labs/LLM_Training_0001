"""Muon Optimizer: 2D パラメータ（重み行列）専用最適化

Reference: https://github.com/KellerJordan/Muon
2D以外のパラメータ（embedding, bias, LayerNorm）は AdamW で別途最適化。
"""

import torch
from torch.optim import Optimizer


def get_optimal_newton_schulz_steps(hidden_size: int = 768, n_params: int = 150_000_000) -> int:
    """モデルスケール（隠れ層次元/パラメータ数）に応じた Newton-Schulz 最適反復数の算出。

    小規模モデル（150M級 / hidden_size < 2048）: 高速化重視で steps = 3
    中・大規模モデル（3B/7B級 / hidden_size >= 2048 または n_params >= 1B）: 直交化精度重視で steps = 5
    """
    if hidden_size >= 2048 or n_params >= 1_000_000_000:
        return 5
    return 3


class Muon(Optimizer):
    """Muon optimizer for 2D parameters (weight matrices).

    Args:
        params: 2Dパラメータのみ（ndim==2）
        lr: 学習率（推奨: max_lr_2d と同一）
        momentum: モメンタム係数（デフォルト: 0.95）
        nesterov: ネステロフモメンタム（デフォルト: True）
        ns_steps: Newton-Schulz 反復数（Noneの場合はモデルサイズから自動動的判定）
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int | None = None,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps or 3)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            ns_steps = group.get("ns_steps", 3)
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

                # Newton-Schulz直交化（JITコンパイル済み関数で高速化）
                update = zeropower_via_newtonschulz(update, steps=ns_steps)

                # パラメータ更新
                p.add_(update, alpha=-group["lr"])


@torch.jit.script
def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 3) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalization (JIT Compiled).

    5x5 Toeplitz coefficients (最適化済み):
    a=3.4445, b=-4.7750, c=2.03153
    """
    a, b, c = 3.4445, -4.7750, 2.03153
    # 数値的オーバーフロー（fp16溢れ）および精度低下を防ぐため、
    # 計算過程のみ float32 にキャストする
    G_fp32 = G.to(torch.float32)
    X = G_fp32 / (G_fp32.norm() + 1e-7)

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)

    if transposed:
        X = X.T

    return X.to(G.dtype)
