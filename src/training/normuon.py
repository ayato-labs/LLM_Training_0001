import torch


def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Newton-Schulz法による行列の直交化（Orthogonalization by Newton-Schulz iteration）。

    与えられた行列Gを直交行列に変換する。
    論文: "Neural Orthogonalization" 等を参照。
    """
    assert G.ndim == 2
    m, n = G.shape
    transposed = m < n
    if transposed:
        G = G.T
        m, n = n, m
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if transposed:
        X = X.T
    return X


def neuron_norm(update: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """行各々の正規化（Row-wise normalization）。"""
    row_norms = update.norm(dim=1, keepdim=True).clamp(min=eps)
    return update / row_norms


class NorMuon(torch.optim.Optimizer):
    """NorMuonオプティマイザ：Newton-Schulz直交化に基づくオプティマイザ。

    2Dパラメータに対して直交化を行いながら学習率を適用する。
    Muonähnlich（Muonに類似した）動作を実現。
    """

    def __init__(
        self, params, lr=0.0235, momentum=0.95, ns_steps=7, weight_decay=0.1, cautious=True
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            cautious=cautious,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """オプティマイザのステップ実行。"""
        for group in self.param_groups:
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            weight_decay, cautious = group["weight_decay"], group["cautious"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "NorMuon supports 2D params only."
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                # Newton-Schulz直交化を実行
                update = newton_schulz_orthogonalize(buf, steps=ns_steps)
                update = neuron_norm(update)
                # 重み減衰の適用
                if weight_decay > 0:
                    if cautious:
                        # 符号が一致する場合のみ減衰（穏やかな適用）
                        mask = (update.sign() == p.sign()).float()
                        p.data.mul_(1 - lr * weight_decay * mask)
                    else:
                        p.data.mul_(1 - lr * weight_decay)
                # パラメータ更新
                p.data.add_(update, alpha=-lr)
