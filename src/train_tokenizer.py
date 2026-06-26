from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from pathlib import Path

# Load corpus
CORPUS_PATH = Path("data/corpus.jsonl")
TOKENIZER_PATH = Path("data/tokenizer.json")

def train_tokenizer():
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=32000)
    
    files = [str(CORPUS_PATH)]
    tokenizer.train(files, trainer)
    
    # Save
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")

if __name__ == "__main__":
    train_tokenizer()
