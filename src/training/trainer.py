from transformers import Trainer
from src.common.logger import logger


class CustomTrainer(Trainer):
    """
    Muon/AdamW分割オプティマイザを自動構築するカスタムTrainer。
    """
    def __init__(self, *args, additional_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_config = additional_config

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.additional_config
        # hpo または training の設定からハイパーパラメータを取得
        hpo_config = config.get("hpo", config)

        lr_2d = hpo_config.get("max_lr_2d", 3e-4)
        lr_1d = hpo_config.get("max_lr_1d", 3e-3)
        weight_decay = hpo_config.get("weight_decay", 0.1)
        beta2 = hpo_config.get("beta2", 0.95)

        # 2Dパラメータ（行列）と1Dパラメータ（embeddings, biases, layernorms）に分類
        params_2d = []
        params_1d = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) < 2 or "embed" in n or "norm" in n or "bias" in n or "lm_head" in n:
                params_1d.append(p)
            else:
                params_2d.append(p)

        try:
            from muon import Muon

            logger.info(f"Optimizer: Muon for 2D (lr={lr_2d}), AdamW for 1D (lr={lr_1d})")
            self.optimizer = Muon(
                params_2d,
                lr=lr_2d,
                momentum=0.95,
                adamw_params=dict(
                    params=params_1d,
                    lr=lr_1d,
                    betas=(0.9, beta2),
                    weight_decay=weight_decay,
                ),
            )
        except ImportError:
            logger.info("Optimizer: Muon not found. Falling back to split AdamW.")
            from torch.optim import AdamW

            self.optimizer = AdamW(
                [
                    {"params": params_2d, "lr": lr_2d, "weight_decay": 0.0},
                    {
                        "params": params_1d,
                        "lr": lr_1d,
                        "weight_decay": weight_decay,
                    },
                ],
                betas=(0.9, beta2),
            )

        return self.optimizer
