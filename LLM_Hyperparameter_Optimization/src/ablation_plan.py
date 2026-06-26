# experiments/ablation_plan.py
"""
IMU-1 Table 5, 6 の追試。
70M Proxy モデルで各変更の単独効果と合計効果を測定。
"""

ABLATION_EXPERIMENTS = [
    # ── Baseline ──
    {
        'name': 'baseline',
        'use_qk_norm': False, 'use_value_residual': False,
        'use_layernorm_scaling': False, 'use_per_head_gating': False,
        'optimizer': 'adamw', 'lr': 0.002,
    },

    # ── アーキテクチャ ablation（単独） ──
    {
        'name': 'qk_norm_only',
        'use_qk_norm': True, 'use_value_residual': False,
        'use_layernorm_scaling': False, 'use_per_head_gating': False,
        'optimizer': 'adamw', 'lr': 0.002,
    },
    {
        'name': 'value_residual_only',
        'use_qk_norm': False, 'use_value_residual': True,
        'use_layernorm_scaling': False, 'use_per_head_gating': False,
        'optimizer': 'adamw', 'lr': 0.002,
    },

    # ── 全アーキテクチャ変更 ──
    {
        'name': 'all_arch',
        'use_qk_norm': True, 'use_value_residual': True,
        'use_layernorm_scaling': True, 'use_per_head_gating': True,
        'optimizer': 'adamw', 'lr': 0.002,
    },

    # ── オプティマイザ ablation ──
    {
        'name': 'all_arch_normuon',
        'use_qk_norm': True, 'use_value_residual': True,
        'use_layernorm_scaling': True, 'use_per_head_gating': True,
        'optimizer': 'normuon', 'lr_2d': 0.0235, 'lr_1d': 0.007,
    },
    {
        'name': 'all_arch_normuon_cwd',   # ★ フル構成
        'use_qk_norm': True, 'use_value_residual': True,
        'use_layernorm_scaling': True, 'use_per_head_gating': True,
        'optimizer': 'normuon_cwd', 'lr_2d': 0.0235, 'lr_1d': 0.007,
    },
]
