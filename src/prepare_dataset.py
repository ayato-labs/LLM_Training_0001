from pathlib import Path

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

# Paths
CORPUS_PATH = Path("data/corpus.jsonl")
TOKENIZER_PATH = Path("data/tokenizer.json")
DATASET_PATH = Path("data/dataset")


def prepare_dataset():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    tokenizer.pad_token = "[PAD]"

    # Load dataset
    dataset = load_dataset("json", data_files=str(CORPUS_PATH))

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "metadata"]
    )

    # Save
    tokenized_dataset.save_to_disk(str(DATASET_PATH))
    print(f"Dataset tokenized and saved to {DATASET_PATH}")


if __name__ == "__main__":
    prepare_dataset()
