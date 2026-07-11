import optuna

from src.logger import logger
from src.step_law import compute_hpo_for_target
from src.trainer import train_model


def objective(trial):
    try:
        hpo_prior = compute_hpo_for_target(70_000_000, 1_400_000_000)

        lr = trial.suggest_float(
            "lr", hpo_prior["max_lr_2d"] * 0.5, hpo_prior["max_lr_2d"] * 2.0, log=True
        )
        logger.debug(f"Trial {trial.number}: lr={lr:.6f}")

        config = {
            "model_name": "modern_gpt",
            "model_config": {"n_layer": 1, "n_embd": 32, "n_head": 4, "vocab_size": 1000},
            "lr": lr,
            "optimizer": "normuon",
        }

        loss = train_model(config, trial=trial)
        logger.info(f"Trial {trial.number}: loss={loss:.6f}")
        return loss
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logger.info("Starting HPO study")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    logger.info(f"Best params: {study.best_params}")
