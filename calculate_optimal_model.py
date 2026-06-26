import sys
import yaml
import logging
from pathlib import Path
from datasets import load_from_disk
from LLM_Hyperparameter_Optimization.src.step_law import compute_hpo_for_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_config_with_hpo(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset to get token count
    dataset_path = Path(config['paths']['dataset_path'])
    dataset = load_from_disk(str(dataset_path))
    # トークン数の概算 (各サンプルが何トークンかによるが、ここではサンプル数 * avg_len と仮定するか、データセットの総トークンを使う)
    # 簡易的にサンプル数 * seq_len としておく
    total_tokens = len(dataset['train']) * config['model']['params'].get('max_position_embeddings', 512)

    # Scaling settings
    target_params = config['scaling_optimization']['target_model_params']
    scale_factor = config['scaling_optimization']['search_scale_factor']
    min_ratio = config['scaling_optimization']['min_data_to_param_ratio']

    # Validation
    ratio = total_tokens / target_params
    if ratio < min_ratio:
        logger.warning(f"Data-to-param ratio is low: {ratio:.2f} (Target: {min_ratio}). Model may overfit.")
    elif ratio > min_ratio * 100:
        logger.warning(f"Data-to-param ratio is very high: {ratio:.2f}. You could train a larger model.")

    # Calculate params for search
    search_params = int(target_params * scale_factor)
    
    hpo = compute_hpo_for_target(
        n_params=search_params,
        n_tokens=total_tokens,
        seq_len=config['model']['params'].get('max_position_embeddings', 512),
    )

    logger.info(f"--- Updating Configuration ---")
    logger.info(f"Target Params: {target_params:,}, Search Params: {search_params:,}")
    logger.info(f"New Max LR: {hpo['max_lr_2d']:.6f}")

    # Update config
    config['training']['learning_rate'] = float(hpo['max_lr_2d'])
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Updated {config_path} with new learning rate based on scaling laws.")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    update_config_with_hpo(config_path)
