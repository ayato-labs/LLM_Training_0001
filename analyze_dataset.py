from datasets import load_from_disk
from pathlib import Path

def count_tokens(dataset_path):
    dataset = load_from_disk(str(dataset_path))
    # input_ids の合計長を計算
    total_tokens = sum(len(sample['input_ids']) for sample in dataset['train'])
    return total_tokens

if __name__ == "__main__":
    total = count_tokens("data/dataset")
    print(f"Total tokens in dataset: {total:,}")
