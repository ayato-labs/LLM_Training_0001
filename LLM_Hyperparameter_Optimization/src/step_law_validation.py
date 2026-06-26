# evaluation/step_law_validation.py

def record_step_law_accuracy(
    n_params: int,
    n_tokens: int,
    predicted_lr: float,
    actual_best_lr: float,
    final_loss: float,
    chinchilla_loss: float,
) -> dict:
    """
    Step Law の予測精度を記録。
    """
    lr_error_pct = abs(predicted_lr - actual_best_lr) / actual_best_lr * 100
    return {
        'n_params'       : n_params,
        'n_tokens'       : n_tokens,
        'predicted_lr'   : predicted_lr,
        'actual_best_lr' : actual_best_lr,
        'lr_error_pct'   : lr_error_pct,
        'final_loss'     : final_loss,
        'chinchilla_loss': chinchilla_loss,
        'data_efficiency': chinchilla_loss / final_loss,
    }
