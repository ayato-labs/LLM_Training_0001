from pathlib import Path
from src.scaling_analysis import get_experiment_data

def run_analysis(models_dir="models"):
    models_path = Path(models_dir)
    experiments = []
    
    for exp_dir in models_path.iterdir():
        if exp_dir.is_dir():
            data = get_experiment_data(exp_dir)
            if data:
                experiments.append(data)
                print(f"Found experiment: {exp_dir.name} -> Params: {data['params']:.2e}, Loss: {data['loss']:.4f}")
    
    if len(experiments) < 2:
        print("Need at least 2 experiments to start scaling analysis.")
    else:
        print(f"Successfully collected {len(experiments)} experiments.")
        # ここで最小二乗法などでフィッティングを行う関数を呼び出す

if __name__ == "__main__":
    run_analysis()
