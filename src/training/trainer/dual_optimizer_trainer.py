"""DualOptimizerTrainer: SplitOptimizer を使用する HF Trainer サブクラス

SplitOptimizer は Muon(2D) + AdamW 8bit(1D) を組み合わせた複合最適化を提供。
2Dパラメータ（重み行列）: Muon
1Dパラメータ（embedding, bias, LayerNorm）: AdamW 8bit
"""

import sys

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer

from src.common.logger import logger
from src.training.optimizers.muon import Muon


class SplitOptimizer:
    """SplitOptimizer: Muon(2D) + AdamW 8bit(1D) の複合最適化

    2Dパラメータ（重み行列）: Muon
    1Dパラメータ（embedding, bias, LayerNorm）: AdamW 8bit
    
    LRスケジューラ対応: param_groups を公開し、LambdaLR が両方に比例適用可能
    """

    def __init__(self, model, config: dict):
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []

        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 2:
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
        except ImportError:
            logger.warning(
                "bitsandbytes not available. Using standard AdamW (higher VRAM usage)."
            )
            self.adamw = torch.optim.AdamW(
                adamw_params,
                lr=config.get("max_lr_1d", 3e-3),
                betas=(0.9, config.get("beta2", 0.95)),
                weight_decay=config.get("weight_decay", 0.1),
            )

        # LRスケジューラ用: 両オプティマイザのparam_groupsを統合公開
        # HF LambdaLR は optimizer.param_groups を更新するため、proxy param_groups を提供
        self._param_groups = []
        for pg in self.muon.param_groups:
            self._param_groups.append({"lr": pg["lr"], "optimizer": "muon", "source_pg": pg})
        for pg in self.adamw.param_groups:
            self._param_groups.append({"lr": pg["lr"], "optimizer": "adamw", "source_pg": pg})

        logger.info(
            f"SplitOptimizer initialized: "
            f"Muon(2D): {len(muon_params)} params, "
            f"AdamW(1D): {len(adamw_params)} params, "
            f"lr_2d={config.get('max_lr_2d', 3e-4):.2e}, "
            f"lr_1d={config.get('max_lr_1d', 3e-3):.2e}"
        )

    @property
    def param_groups(self):
        """HF LRスケジューラ互換: 両オプティマイザのparam_groupsを統合して公開"""
        return self._param_groups

    def _sync_param_groups(self):
        """内部オプティマイザのLRを_proxy param_groupsから同期"""
        for pg in self._param_groups:
            pg["source_pg"]["lr"] = pg["lr"]

    def step(self):
        # LR同期してからstep
        self._sync_param_groups()
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "_param_groups": self._param_groups,  # LR状態も保存
        }

    def load_state_dict(self, state: dict):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        if "_param_groups" in state:
            self._param_groups = state["_param_groups"]


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
