import optuna
from .trainer import train_model
from .step_law import compute_hpo_for_target

def objective(trial):
    # Step Law から最適 LR の目安を取得
    # 70M モデル、1.4B トークンをターゲットにする
    hpo_prior = compute_hpo_for_target(70_000_000, 1_400_000_000)
    
    # Optuna で探索（Step Law の予測値を中心に探索）
    lr = trial.suggest_float("lr", hpo_prior['max_lr_2d'] * 0.5, hpo_prior['max_lr_2d'] * 2.0, log=True)
    
    config = {
        'model_name': 'modern_gpt',
        'model_config': {
            'n_layer': 1,
            'n_embd': 32,
            'n_head': 4,
            'vocab_size': 1000
        },
        'lr': lr,
        'optimizer': 'normuon'
    }
    
    loss = train_model(config, trial=trial)
    return loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("Best params:", study.best_params)
