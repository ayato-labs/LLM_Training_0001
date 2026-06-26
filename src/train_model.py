import torch
import json
import hashlib
import sys
import shutil
import subprocess
import datetime
from pathlib import Path
from datasets import load_dataset
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

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def train(config_path):
    config = load_config(config_path)
    git_hash = get_git_revision_hash()
    
    # データのロード
    print("Starting dataset load...")
    dataset = load_dataset("json", data_files=config['data_path'])
    print(f"Dataset loaded. Size: {len(dataset['train'])}")
    
    # トークナイザー設定 (仮: data/tokenizer.json があると仮定)
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    print("Tokenizer loaded.")

    # モデル初期化
    print("Initializing model...")
    model_config = LlamaConfig(
        **config['model_params'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = LlamaForCausalLM(model_config)
    print("Model initialized.")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    print("Initializing trainer...")
    training_args = TrainingArguments(
        output_dir="models/output",
        learning_rate=config['hpo']['max_lr_2d'],
        per_device_train_batch_size=config['hpo']['batch_size_seqs'],
        num_train_epochs=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    
    print("Trainer initialized. Starting training...", flush=True)
    trainer.train()
    
    model.save_pretrained("models/output")
    tokenizer.save_pretrained("models/output")
    print("Training finished.")

if __name__ == "__main__":
    config_path = sys.argv[1]
    train(config_path)
