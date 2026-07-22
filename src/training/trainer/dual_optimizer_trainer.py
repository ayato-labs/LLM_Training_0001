"""DualOptimizerTrainer: SplitOptimizer を使用する HF Trainer サブクラス

SplitOptimizer は Muon(2D) + AdamW 8bit(1D) を組み合わせた複合最適化を提供。
2Dパラメータ（重み行列）: Muon
1Dパラメータ（embedding, bias, LayerNorm）: AdamW 8bit
"""

import torch
from torch.optim import Optimizer
from transformers import Trainer

from src.common.logger import logger
from src.training.optimizers.muon import Muon


class SplitOptimizer(Optimizer):
    """SplitOptimizer: Muon(2D) + AdamW 8bit(1D) の複合最適化

    2Dパラメータ（重み行列）: Muon
    1Dパラメータ（embedding, bias, LayerNorm）: AdamW 8bit

    torch.optim.Optimizer を継承し、LambdaLR 等の標準スケジューラと互換。
    """

    def __init__(self, model, config: dict):
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []

        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon should only optimize internal 2D weight matrices (not embedding or lm_head)
            if param.ndim == 2 and not any(x in _name for x in ["embed_tokens", "lm_head"]):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.muon = Muon(muon_params, lr=config.get("max_lr_2d", 3e-4))

        # AdamW 8bit (VRAM節約)
        try:
            import bitsandbytes as bnb

            self.adamw = bnb.optim.AdamW8bit(
                adamw_params,
                lr=config.get("max_lr_1d", 3e-3),
                betas=(0.9, config.get("beta2", 0.95)),
                weight_decay=config.get("weight_decay", 0.1),
            )
            logger.info("Successfully initialized bitsandbytes 8-bit AdamW for 1D parameters.")
        except Exception as e:
            logger.warning(
                f"bitsandbytes 8-bit AdamW unavailable ({e}). "
                "Falling back to standard FP32 AdamW (higher VRAM usage)."
            )
            self.adamw = torch.optim.AdamW(
                adamw_params,
                lr=config.get("max_lr_1d", 3e-3),
                betas=(0.9, config.get("beta2", 0.95)),
                weight_decay=config.get("weight_decay", 0.1),
            )

        # Optimizer基底クラス初期化 (param_groups で LRスケジューラ対応)
        # muon + adamw の param_groups を統合
        param_groups = []
        for pg in self.muon.param_groups:
            param_groups.append(pg)
        for pg in self.adamw.param_groups:
            param_groups.append(pg)

        defaults = dict(lr=config.get("max_lr_2d", 3e-4))  # base lr
        super().__init__(param_groups, defaults)

        logger.info(
            f"SplitOptimizer initialized: "
            f"Muon(2D): {len(muon_params)} params, "
            f"AdamW(1D): {len(adamw_params)} params, "
            f"lr_2d={config.get('max_lr_2d', 3e-4):.2e}, "
            f"lr_1d={config.get('max_lr_1d', 3e-3):.2e}"
        )

    def _sync_param_groups(self):
        """内部オプティマイザのLRを param_groups から同期"""
        for i, pg in enumerate(self.param_groups):
            if i < len(self.muon.param_groups):
                self.muon.param_groups[i]["lr"] = pg["lr"]
            else:
                j = i - len(self.muon.param_groups)
                self.adamw.param_groups[j]["lr"] = pg["lr"]

    def step(self, closure=None):
        # LR同期してからstep
        self._sync_param_groups()
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


class DualOptimizerTrainer(Trainer):
    """SplitOptimizer を使用するカスタムTrainer

    既存Trainerのgradient accumulation, mixed precision, checkpointing を
    そのまま利用しつつ、Muon(2D) + AdamW(1D) の分離最適化を実現。
    """

    def __init__(self, *args, split_optimizer_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_optimizer_config = split_optimizer_config or {}

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = SplitOptimizer(
                self.model,
                self.split_optimizer_config,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """SplitOptimizer対応スケジューラ作成

        SplitOptimizerのparam_groups (muon + adamw統合) に対して
        単一のLambdaLRを適用。比率 max_lr_2d : max_lr_1d を保持。
        """
        from src.training.schedulers import create_scheduler_from_config

        opt = optimizer or self.optimizer
        config = self.split_optimizer_config

        # 設定からスケジューラ生成
        scheduler = create_scheduler_from_config(opt, config, num_training_steps)

        if scheduler is not None:
            self.lr_scheduler = scheduler
            return scheduler

        # フォールバック: 親クラスのデフォルト (cosine)
        return super().create_scheduler(num_training_steps, opt)
