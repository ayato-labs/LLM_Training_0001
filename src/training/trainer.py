from transformers import Trainer
from src.common.logger import logger


class CustomTrainer(Trainer):
    """
    Muon/AdamW分割オプティマイザを自動構築するカスタムTrainer。
    
    モデルのパラメータ特性（重み行列の次元数など）に応じて、
    最適な最適化アルゴリズム（Muon または AdamW）を割り当てるためのカスタムクラス。
    """
    def __init__(self, *args, additional_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_config = additional_config

    def create_optimizer(self):
        """
        カスタムオプティマイザの初期化処理。
        
        Muonオプティマイザ（インポート可能な場合）を2Dパラメータに適用し、
        1Dパラメータ（埋め込み、バイアス、LayerNormなど）にはAdamWを適用する。
        Muonが利用できない場合は、ハイパーパラメータを分割したAdamWにフォールバックする。
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        config = self.additional_config
        # HPO (ハイパーパラメータ最適化) または通常の学習設定からパラメータ値を取得
        hpo_config = config.get("hpo", config)

        lr_2d = hpo_config.get("max_lr_2d", 3e-4)
        lr_1d = hpo_config.get("max_lr_1d", 3e-3)
        weight_decay = hpo_config.get("weight_decay", 0.1)
        beta2 = hpo_config.get("beta2", 0.95)

        # パラメータの分類:
        # - 2Dパラメータ（主に通常の線形層・射影層などの重み行列。Muonによる直交化更新が効果的な対象）
        # - 1Dパラメータ（1次元テンソル、バイアス、埋め込み層、LayerNormのスケール因子。AdamWで最適化する対象）
        params_2d = []
        params_1d = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 1次元未満のパラメータ、あるいは埋め込み(embed)、レイヤー正規化(norm)、バイアス(bias)、出力ヘッド(lm_head)は1Dとして分類
            if len(p.shape) < 2 or "embed" in n or "norm" in n or "bias" in n or "lm_head" in n:
                params_1d.append(p)
            else:
                params_2d.append(p)

        try:
            # Muonオプティマイザのロードを試行
            from muon import Muon

            logger.info(f"Optimizer: Muon for 2D (lr={lr_2d}), AdamW for 1D (lr={lr_1d})")
            # 2D行列に対してはMuonを使用し、1Dパラメータ群に対しては内部でAdamWを適用
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
            # Muonが環境にインストールされていない場合は、通常のAdamWによるグループ分割最適化へフォールバック
            logger.info("Optimizer: Muon not found. Falling back to split AdamW.")
            from torch.optim import AdamW

            # 2Dパラメータにはウェイトディケイを適用せず（lr_2d）、1Dパラメータには適用する設定でグループ化
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
