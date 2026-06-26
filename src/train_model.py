import torch
import yaml
import hashlib
import sys
import shutil
import subprocess
import datetime
from pathlib import Path
from datasets import load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizerFast, DataCollatorForLanguageModeling

# ログ設定
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

# Paths
# Now loaded from config
# DATASET_PATH = Path("data/dataset")
# TOKENIZER_PATH = Path("data/tokenizer.json")
# OUTPUT_DIR = Path("models/novel-llm-llama")

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def get_model_class(model_type):
    if model_type == "llama":
        return LlamaConfig, LlamaForCausalLM
    raise ValueError(f"Unsupported model type: {model_type}")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dataset_hash(dataset):
    return hashlib.md5(str(dataset['train'][:100]).encode()).hexdigest()

def save_metadata(output_dir, config_path, dataset_hash, git_hash):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config
    shutil.copy(config_path, output_dir / "config.yaml")
    
    # Save meta
    with open(output_dir / "dataset_meta.txt", "w") as f:
        f.write(f"dataset_hash: {dataset_hash}\n")
        f.write(f"git_hash: {git_hash}\n")

def train(config_path):
    yaml_config = load_config(config_path)
    git_hash = get_git_revision_hash()
    
    # Paths from config
    dataset_path = Path(yaml_config['paths']['dataset_path'])
    tokenizer_path = Path(yaml_config['paths']['tokenizer_path'])
    output_dir_base = Path(yaml_config['paths'].get('output_dir', 'models/output'))
    
    print("Starting dataset load...")
    dataset = load_from_disk(str(dataset_path))
    dataset_hash = get_dataset_hash(dataset)
    print(f"Dataset loaded. Size: {len(dataset['train'])}")
    print(f"Dataset hash: {dataset_hash}")
    print(f"Git commit: {git_hash}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.pad_token = yaml_config['tokenizer']['pad_token']
    tokenizer.bos_token = yaml_config['tokenizer']['bos_token']
    tokenizer.eos_token = yaml_config['tokenizer']['eos_token']
    print("Tokenizer loaded.")

    # Model initialization via factory
    print("Initializing model...")
    config_cls, model_cls = get_model_class(yaml_config['model']['type'])
    
    config = config_cls(
        **yaml_config['model']['params'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    model = model_cls(config)
    print("Model initialized.")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print("Initializing trainer...")
    # Use output_dir from training config or fall back to paths config
    output_dir = yaml_config['training'].pop('output_dir', str(output_dir_base))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        **yaml_config['training']
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    print("Trainer initialized. Starting training...", flush=True)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        raise e
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    save_metadata(output_dir, config_path, dataset_hash, git_hash)
    
    print(f"Training finished. Metadata saved to {output_dir}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    train(config_path)

