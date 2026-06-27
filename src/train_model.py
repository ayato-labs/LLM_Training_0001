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
    
    # トークナイザー設定
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    print("Tokenizer loaded.")
    
    # データのロード
    print("Starting dataset load...")
    dataset = load_dataset("json", data_files=config['data_path'])
    
    # テキストのトークン化
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # 不必要な列を削除
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "metadata"])
    # 形式を整える
    tokenized_datasets.set_format("torch")
    print(f"Dataset loaded. Size: {len(tokenized_datasets['train'])}")
    
    # モデル初期化
    print("Initializing model...")
    # config['model_params'] から hidden_size を取り除いて渡す
    params = config['model_params'].copy()
    params.pop('hidden_size', None)
    
    # hidden_size が num_attention_heads の倍数になるように保証
    hidden_size = config['model_params']['hidden_size']
    num_heads = config['model_params']['num_attention_heads']
    adjusted_hidden_size = (hidden_size // num_heads) * num_heads
    
    model_config = LlamaConfig(
        **params,
        hidden_size=adjusted_hidden_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = LlamaForCausalLM(model_config)
    print(f"Model initialized with hidden_size={adjusted_hidden_size}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 設定から学習ステップ数を取得（指定がない場合は1エポック）
    max_steps = config.get('max_steps', -1)
    
    print("Initializing trainer...")
    # 明示的に None ではなく 0 や -1 を設定して検証を回避
    num_epochs = 1 if max_steps == -1 else 0
    
    training_args = TrainingArguments(
        output_dir="models/output",
        learning_rate=config['hpo']['max_lr_2d'],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps != -1 else -1,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
    )
    
    print("Trainer initialized. Starting training...", flush=True)
    train_result = trainer.train()
    
    # 学習結果の記録
    with open("last_run_result.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f)
    
    model.save_pretrained("models/output")
    tokenizer.save_pretrained("models/output")
    print("Training finished.")

if __name__ == "__main__":
    config_path = sys.argv[1]
    train(config_path)
