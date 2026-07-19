"""LR Schedulers: Step Law推奨 Constant+Cosine スケジューラ

Step Law (arXiv:2503.04715) 推奨:
- 定数LRウォームアップ (数step) → 定数LR維持 → Cosine減衰
- min_lr = 1e-5 固定
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_constant_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_constant_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Constant + Cosine LRスケジューラ
    
    フェーズ:
    1. Warmup (0 → base_lr): num_warmup_steps
    2. Constant (base_lr): num_constant_steps  
    3. Cosine Decay (base_lr → min_lr): 残りステップ
    
    Args:
        optimizer: オプティマイザ
        num_warmup_steps: ウォームアップステップ数
        num_constant_steps: 定数LR維持ステップ数
        num_training_steps: 総ステップ数
        min_lr_ratio: min_lr = base_lr * min_lr_ratio (Step Law推奒: 1e-5 / base_lr)
        last_epoch: 再開時のエポック
    """
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Phase 1: Linear Warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        elif current_step < num_warmup_steps + num_constant_steps:
            # Phase 2: Constant LR
            return 1.0
        
        else:
            # Phase 3: Cosine Decay
            progress = float(current_step - num_warmup_steps - num_constant_steps)
            decay_steps = max(1, num_training_steps - num_warmup_steps - num_constant_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress / decay_steps))
            return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_step_law_schedule(
    optimizer: Optimizer,
    max_lr: float,
    num_warmup_steps: int,
    num_constant_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Step Law標準スケジューラ (min_lr=1e-5固定)
    
    Args:
        optimizer: オプティマイザ
        max_lr: 最大学習率 (Step Law算出値)
        num_warmup_steps: ウォームアップステップ (Step Law推奨: 2-10)
        num_constant_steps: 定数LRステップ
        num_training_steps: 総ステップ
        min_lr: 最小学習率 (Step Law推奨: 1e-5)
    """
    min_lr_ratio = min_lr / max_lr
    return get_constant_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_constant_steps=num_constant_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
        last_epoch=last_epoch,
    )


class StepLawLRScheduler:
    """
    Step Law LRスケジューラのラッパー
    DualOptimizer (Muon + AdamW) 両方に同一スケジュール適用
    """
    
    def __init__(
        self,
        optimizer,
        max_lr_2d: float,
        max_lr_1d: float,
        num_warmup_steps: int,
        num_constant_steps: int,
        num_training_steps: int,
        min_lr: float = 1e-5,
    ):
        self.max_lr_2d = max_lr_2d
        self.max_lr_1d = max_lr_1d
        self.min_lr = min_lr
        
        # Muon用スケジューラ
        self.scheduler_2d = get_step_law_schedule(
            optimizer=optimizer.muon if hasattr(optimizer, 'muon') else optimizer,
            max_lr=max_lr_2d,
            num_warmup_steps=num_warmup_steps,
            num_constant_steps=num_constant_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )
        
        # AdamW用スケジューラ
        self.scheduler_1d = get_step_law_schedule(
            optimizer=optimizer.adamw if hasattr(optimizer, 'adamw') else optimizer,
            max_lr=max_lr_1d,
            num_warmup_steps=num_warmup_steps,
            num_constant_steps=num_constant_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )
    
    def step(self):
        """両スケジューラを同時にステップ"""
        self.scheduler_2d.step()
        self.scheduler_1d.step()
    
    def get_last_lr(self):
        """現在のLR取得 (2D, 1D)"""
        return {
            "lr_2d": self.scheduler_2d.get_last_lr()[0] if self.scheduler_2d.get_last_lr() else 0,
            "lr_1d": self.scheduler_1d.get_last_lr()[0] if self.scheduler_1d.get_last_lr() else 0,
        }
    
    def state_dict(self):
        return {
            "scheduler_2d": self.scheduler_2d.state_dict(),
            "scheduler_1d": self.scheduler_1d.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.scheduler_2d.load_state_dict(state_dict["scheduler_2d"])
        self.scheduler_1d.load_state_dict(state_dict["scheduler_1d"])


def create_scheduler_from_config(
    optimizer,
    config: dict,
    num_training_steps: int,
):
    """HF Trainer互換のスケジューラファクトリ
    
    Args:
        optimizer: SplitOptimizer (muon + adamw)
        config: hpo_config dict with keys:
            - lr_scheduler_type: "step_law" | "cosine" | "constant_cosine"
            - warmup_steps: int
            - warmup_ratio: float
            - constant_steps: int
            - constant_ratio: float
            - min_lr: float
            - num_cycles: float
        num_training_steps: 総ステップ数
    """
    scheduler_type = config.get("lr_scheduler_type", "step_law")
    
    if scheduler_type == "step_law":
        # StepLawLRScheduler は SplitOptimizer (muon + adamw) を直接受け取る
        return StepLawLRScheduler(
            optimizer=optimizer,
            max_lr_2d=config.get("max_lr_2d", 3e-4),
            max_lr_1d=config.get("max_lr_1d", 3e-3),
            num_warmup_steps=config.get("warmup_steps", 0),
            num_constant_steps=config.get("constant_steps", 0),
            num_training_steps=num_training_steps,
            min_lr=config.get("min_lr", 1e-5),
        )
    
    elif scheduler_type in ("constant_cosine", "cosine"):
        # LambdaLRベースのスケジューラ (SplitOptimizerのparam_groupsに適用)
        warmup_steps = config.get("warmup_steps", 0)
        warmup_ratio = config.get("warmup_ratio", 0.0)
        if warmup_steps == 0 and warmup_ratio > 0:
            warmup_steps = int(num_training_steps * warmup_ratio)
        
        constant_steps = config.get("constant_steps", 0)
        constant_ratio = config.get("constant_ratio", 0.0)
        if constant_steps == 0 and constant_ratio > 0:
            constant_steps = int(num_training_steps * constant_ratio)
        
        min_lr_ratio = config.get("min_lr_ratio", 0.0)
        if min_lr_ratio == 0.0 and config.get("min_lr", 0) > 0:
            max_lr = config.get("max_lr_2d", 1.0)
            min_lr_ratio = config["min_lr"] / max_lr
        
        num_cycles = config.get("num_cycles", 0.5)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            if current_step < warmup_steps + constant_steps:
                return 1.0
            
            progress = float(current_step - warmup_steps - constant_steps)
            decay_steps = max(1, num_training_steps - warmup_steps - constant_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress / decay_steps))
            return max(min_lr_ratio, cosine_decay)
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {scheduler_type}")


def get_recommended_scheduler_config(
    max_steps: int,
    warmup_ratio: float = 0.03,
    constant_ratio: float = 0.1,
    scheduler_type: str = "constant_cosine",
) -> dict:
    """Step Law / Muon 推奨のデフォルト設定を返す"""
    return {
        "lr_scheduler_type": scheduler_type,
        "warmup_steps": int(max_steps * warmup_ratio),
        "constant_steps": int(max_steps * constant_ratio),
        "num_cycles": 0.5,
    }