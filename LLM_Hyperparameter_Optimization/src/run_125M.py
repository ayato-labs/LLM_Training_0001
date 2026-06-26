from step_law import compute_hpo_for_target

# Step Law で HPO を計算
hpo = compute_hpo_for_target(
    n_params=125_000_000,
    n_tokens=10_000_000_000,
    seq_len=512,
)

FULL_RUN_125M = {
    'n_embd'          : 768,
    'n_layer'         : 12,
    'n_head'          : 12,
    'n_kv_head'       : 4,
    'block_size'      : 512,
    'all_arch_mods'   : True,
    'lr_2d'           : hpo['max_lr_2d'],
    'lr_1d'           : hpo['max_lr_1d'],
    'stable_lr_2d'    : hpo['stable_lr_2d'],
    'min_lr'          : 1e-5,
    'schedule'        : 'WSD',
    'warmup_frac'     : 0.01,
    'decay_frac'      : 0.20,
    'decay_profile'   : '1-sqrt',
    'batch_size_seqs' : 8,
    'grad_accum_steps': hpo['batch_size_seqs'] // 8,
    'dataset'         : 'FineWeb-edu',
    'total_tokens'    : 10_000_000_000,
    'use_ema'         : True,
    'ema_beta'        : 0.8,
    'ema_n_ckpts'     : 10,
}

if __name__ == '__main__':
    print("HPO Configuration for 125M run:")
    print(FULL_RUN_125M)
