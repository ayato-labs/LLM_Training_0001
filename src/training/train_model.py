import torch
import json
import hashlib
import sys
import shutil
import os
import subprocess
import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import training_config as prj_config

# ログ設定
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def is_deepspeed_available():
    try:
        import deepspeed
        import importlib.metadata
        importlib.metadata.version("deepspeed")
        return True
    except Exception:
        return False

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

class CustomTrainer(Trainer):
    def __init__(self, *args, additional_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_config = additional_config

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        
        model = self.model
        config = self.additional_config
        
        lr_2d = config['hpo']['max_lr_2d']
        lr_1d = config['hpo']['max_lr_1d']
        
        params_2d = []
        params_1d = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 1D/AdamW target parameters (Embeddings, Layernorms, Biases, classifier head)
            if len(p.shape) < 2 or "embed" in n or "norm" in n or "bias" in n or "lm_head" in n:
                params_1d.append(p)
            else:
                params_2d.append(p)
                
        try:
            from muon import Muon
            print(f"Optimizer: Using Muon for 2D params (lr={lr_2d}) and AdamW for 1D params (lr={lr_1d})", flush=True)
            self.optimizer = Muon(
                params_2d,
                lr=lr_2d,
                momentum=0.95,
                adamw_params=dict(
                    params=params_1d,
                    lr=lr_1d,
                    betas=(0.9, 0.95),
                    weight_decay=0.01
                )
            )
        except ImportError:
            print("Optimizer: Muon not found. Falling back to split AdamW optimizer.", flush=True)
            from torch.optim import AdamW
            self.optimizer = AdamW([
                {'params': params_2d, 'lr': lr_2d, 'weight_decay': 0.0},
                {'params': params_1d, 'lr': lr_1d, 'weight_decay': 0.01}
            ], betas=(0.9, 0.95))
            
        return self.optimizer

def generate_deepspeed_config(n_params, vram_limit_gb):
    # 2 bytes weight (fp16) + 12 bytes optimizer states per parameter
    est_vram_gb = (n_params * 14) / (1024**3)
    
    # プレトレーニングにおける極端な速度低下を防ぐため、CPUオフロードは常時無効化（GPUオンリー）
    offload_device = "none"
    
    ds_config = {
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_of_nodes": 1,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "gradient_accumulation_steps": "auto",
        "train_batch_size": "auto"
    }
    
    if est_vram_gb > vram_limit_gb:
        print(f"DeepSpeed WARNING: Estimated parameters memory ({est_vram_gb:.2f} GB) exceeds VRAM limit ({vram_limit_gb} GB). CPU Offload is disabled by design. Running entirely on GPU may trigger OOM (Out Of Memory).", flush=True)
    else:
        print(f"DeepSpeed: Estimated VRAM ({est_vram_gb:.2f} GB) fits within VRAM limit ({vram_limit_gb} GB). Running entirely on GPU.", flush=True)
        
    ds_config_path = "temp_ds_config.json"
    with open(ds_config_path, "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)
    return ds_config_path

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
    data_path = Path(config['data_path'])
    if not data_path.exists():
        fallback_path = Path("data") / data_path.name
        if fallback_path.exists():
            data_path = fallback_path
            print(f"Dataset path resolved to fallback: {data_path}")
        else:
            raise FileNotFoundError(f"Could not find dataset at '{config['data_path']}' or '{fallback_path.resolve()}'")
            
    dataset = load_dataset("json", data_files=str(data_path))
    
    # テキストのトークン化
    seq_len = config['hpo'].get('seq_len', 1024)
    print(f"Tokenizing dataset with max_length={seq_len}...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_len)
    
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
    num_epochs = getattr(prj_config, "NUM_EPOCHS", 3) if max_steps == -1 else 0
    
    # スケーリングに対応した動的バッチサイズ・DeepSpeed設定の生成
    hpo_config = config['hpo']
    target_total_batch_seqs = hpo_config.get('batch_size_seqs', 16)
    
    # VRAMサイズとモデル規模からDeepSpeed構成を決定
    # パラメータ数が指定されていなければデフォルト125Mとする
    n_params_est = config['model_params'].get('n_params', 125_000_000)
    vram_limit = prj_config.VRAM_LIMIT_GB
    ds_config_path = generate_deepspeed_config(n_params_est, vram_limit)
    
    # 適切な1デバイスあたりバッチサイズと勾配蓄積ステップの計算
    # WindowsのWDDMメモリページングによる極端な速度低下を防ぐため、物理VRAMサイズに応じて最大バッチサイズを動的に制限する。
    if vram_limit <= 4.5:
        max_batch = 1  # 4GB VRAM環境下では最大1バッチに変更しWDDMページングを回避
    elif vram_limit <= 8.5:
        max_batch = 4  # 8GB VRAM
    elif vram_limit <= 16.5:
        max_batch = 8  # 16GB VRAM
    else:
        max_batch = 16
        
    per_device_batch = min(target_total_batch_seqs, max_batch)
    grad_accum_steps = max(1, target_total_batch_seqs // per_device_batch)
    
    warmup_ratio = hpo_config.get('warmup_ratio', 0.03)

    training_args = TrainingArguments(
        output_dir="models/output",
        learning_rate=hpo_config['max_lr_2d'],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps != -1 else -1,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        deepspeed=ds_config_path if (torch.cuda.is_available() and is_deepspeed_available()) else None,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=10,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
        additional_config=config,
    )
    
    print(f"Trainer initialized with batch_size={per_device_batch}, grad_accum={grad_accum_steps}, scheduler=cosine (warmup={warmup_ratio})", flush=True)
    print("Starting training...", flush=True)
    train_result = trainer.train()
    
    # 学習結果の記録
    with open("last_run_result.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f)
    
    # 一時的なDeepSpeed設定ファイルをクリーンアップ
    if os.path.exists(ds_config_path):
        try:
            os.remove(ds_config_path)
        except:
            pass
            
    model.save_pretrained("models/output")
    tokenizer.save_pretrained("models/output")
    print("Training finished.")

if __name__ == "__main__":
    config_path = sys.argv[1]
    train(config_path)
