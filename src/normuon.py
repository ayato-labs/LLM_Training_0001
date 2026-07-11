import torch

def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 7) -> torch.Tensor:
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
    row_norms = update.norm(dim=1, keepdim=True).clamp(min=eps)
    return update / row_norms

class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0235, momentum=0.95,
                 ns_steps=7, weight_decay=0.1, cautious=True):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay, cautious=cautious)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, ns_steps = group['lr'], group['momentum'], group['ns_steps']
            weight_decay, cautious = group['weight_decay'], group['cautious']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                assert g.ndim == 2, "NorMuon supports 2D params only."
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                update = newton_schulz_orthogonalize(buf, steps=ns_steps)
                update = neuron_norm(update)
                if weight_decay > 0:
                    if cautious:
                        mask = (update.sign() == p.sign()).float()
                        p.data.mul_(1 - lr * weight_decay * mask)
                    else:
                        p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
