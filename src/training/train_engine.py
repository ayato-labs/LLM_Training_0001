import json
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.common.logger import logger
from src.common.set_seed import set_seed
from src.common.env_snapshot import capture_env_snapshot
from src.training.model_utils import (
    create_model_config,
    TokenizerWrapper,
    PackedDatasetWrapper,
    get_optimal_num_proc,
    compute_file_hash,
    compute_dataset_fingerprint,
    compute_db_fingerprint,
)
from src.training.callbacks import (
    ProgressBarFormatCallback,
    HashSaveCallback,
    DetailedLoggingCallback,
)


def train(config: dict, tokenized_datasets=None, extra_callbacks=None):
    """
    統一された学習オーケストレーションフロー。
    
    Args:
        config (dict): 学習設定パラメータを含む辞書。
        tokenized_datasets (dict, optional): トークン化済みのデータセット辞書（学習/検証）。省略時はロード及びトークン化を行う。
        extra_callbacks (list, optional): 追加のTrainerCallbackリスト。
    """
    # 1. 乱数シードの設定（再現性確保のため決定論的挙動を強制）
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True)

    # 1.5 TF32 (TensorFloat-32) の有効化（Ampere世代以降のGPUでの行列演算高速化）
    if config.get("allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cudnn.")

    # 2. 実行環境スナップショットの取得とロギング
    env_snapshot = capture_env_snapshot()
    logger.debug(f"Env snapshot: {env_snapshot}")

    # 3. 現在の設定ファイルとデータセットのハッシュ値の算出（整合性チェック用）
    config_path = Path("configs/config.yaml")
    data_path_str = config.get("data_path", "data/dataset.jsonl")
    current_config_hash = compute_file_hash(str(config_path))
    current_data_hash = compute_file_hash(data_path_str)

    # 4. データセットおよびデータベースのフィンガープリントを取得
    data_fingerprint = compute_dataset_fingerprint(data_path_str)
    db_path_str = config.get("db_path", "../Novel_Data_Collection/novels.db")
    db_fingerprint = compute_db_fingerprint(db_path_str)

    logger.info(
        "Training initialization",
        extra={
            "seed": seed,
            "data_path": data_path_str,
            "dataset_fingerprint": data_fingerprint,
            "db_fingerprint": db_fingerprint,
        },
    )

    # 5. チェックポイントからの再開（Resume）処理の解決
    resume_checkpoint = config.get("resume_from_checkpoint") or config.get("resume")
    
    # "checkpoint-latest" が指定された場合、または True の場合は最新のチェックポイントを自動探索
    if resume_checkpoint is True or (isinstance(resume_checkpoint, str) and "checkpoint-latest" in resume_checkpoint):
        from src.training.model_utils import get_checkpoints
        checkpoints = get_checkpoints()
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1][1])
            logger.info(f"Resolved checkpoint-latest to: {resume_checkpoint}")
        else:
            resume_checkpoint = None
            logger.warning("No checkpoint found to resume from. Starting from scratch.")

    # 6. チェックポイントデータの整合性検証（ハッシュ値による構成変更チェック）
    if isinstance(resume_checkpoint, str) and Path(resume_checkpoint).exists():
        checkpoint_path = Path(resume_checkpoint)
        hash_file = checkpoint_path / "hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    saved_hashes = json.load(f)
                saved_config_hash = saved_hashes.get("config_hash")
                saved_data_hash = saved_hashes.get("data_hash")
                
                # チェックポイント保存時と現在の設定・データが異なる場合はエラーとして学習を中断する
                if saved_config_hash != current_config_hash or saved_data_hash != current_data_hash:
                    logger.error("Configuration or training dataset has changed since the checkpoint was saved!")
                    logger.error(f"Saved Config Hash: {saved_config_hash} | Current Config Hash: {current_config_hash}")
                    logger.error(f"Saved Data Hash: {saved_data_hash} | Current Data Hash: {current_data_hash}")
                    raise ValueError("Cannot resume training: config.yaml or training dataset does not match the checkpoint.")
                else:
                    logger.info("Configuration and dataset hashes match. Verification successful.")
            except Exception as e:
                logger.error(f"Failed to verify checkpoint hashes: {e}")
                raise
        else:
            logger.warning(f"No hashes.json found in checkpoint {resume_checkpoint}. Proceeding without verification.")
    else:
        if not isinstance(resume_checkpoint, str):
            resume_checkpoint = None

    # 7. トークナイザーの読み込み（高速なRust実装版を優先）
    tokenizer_path = Path(config.get("tokenizer_path", "data/tokenizer.json"))
    if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # 特殊トークンの明示的な設定
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    # 8. データセットのロードとトークン化処理（前処理が渡されていない場合のみ実施）
    if tokenized_datasets is None:
        logger.info("Starting dataset loading...")
        data_files = {"train": data_path_str}
        if config.get("val_data_path"):
            data_files["validation"] = config["val_data_path"]

        # JSONLファイルからデータセットをロード
        ds = load_dataset("json", data_files=data_files)
        remove_columns = [c for c in ds["train"].column_names if c in {"text", "metadata"}]

        packing = config.get("packing", False)
        seq_len = config.get("seq_len", 1024)
        tokenize_fn = TokenizerWrapper(tokenizer, seq_len, padding=not packing)
        
        # 利用可能なCPUコア数とメモリ空き容量から、最適な並列プロセス数を動的に算出
        num_proc = get_optimal_num_proc()
        logger.info(f"Tokenizing dataset with num_proc={num_proc} (calculated dynamically from available RAM and CPUs)")

        # マルチプロセスでのトークン化マップ処理の実行
        ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_columns, num_proc=num_proc)

        # Sequence Packingの適用
        if packing:
            logger.info("Packing dataset enabled. Eliminating pad tokens...")
            packed_ds = {}
            for split in ds:
                wrapper = PackedDatasetWrapper(ds[split], seq_len, tokenizer.eos_token_id)
                packed_ds[split] = wrapper()
            from datasets import DatasetDict
            ds = DatasetDict(packed_ds)

        # デバッグや軽量テスト用途のデータ一部抽出 (data_fraction < 1.0 の場合)
        data_fraction = config.get("data_fraction", 1.0)
        if data_fraction < 1.0:
            for split in ds:
                n = int(len(ds[split]) * data_fraction)
                n = max(1, n)
                ds[split] = ds[split].select(range(n))
                logger.info(f"Split '{split}' sampled to {n} samples (fraction={data_fraction})")

        ds.set_format(type="torch")
        train_ds = ds["train"]
        eval_ds = ds.get("validation")
    else:
        # 外部から提供されたトークン化済みデータを設定
        train_ds = tokenized_datasets["train"]
        eval_ds = tokenized_datasets.get("validation")

    # 9. モデルの設定と初期化 (Llamaアーキテクチャ)
    model_config = create_model_config(config, tokenizer)
    model = LlamaForCausalLM(model_config)
    
    # トークナイザーの語彙数に合わせて埋め込み層のサイズを調整
    model.resize_token_embeddings(len(tokenizer))

    # 10. TrainingArguments (学習パラメータ) の構築
    precision = config.get("precision", "bf16")
    output_dir = config.get("output_dir", "models/output")
    hpo_config = config.get("hpo", config)

    max_steps = config.get("max_steps", -1)
    num_epochs = config.get("num_epochs", 3) if max_steps == -1 else 0

    per_device_batch = config.get("per_device_batch_size", 1)
    grad_accum_steps = config.get("grad_accum_steps", 1)
    
    # HPO(ハイパーパラメータ最適化)設定が含まれており、バッチサイズ自動決定が必要な場合の処理
    if "batch_size_seqs" in hpo_config and "per_device_batch_size" not in config:
        target_total_batch_seqs = hpo_config.get("batch_size_seqs", 16)
        vram_limit = config.get("vram_limit_gb", 4.0)
        
        # 利用可能なVRAM容量に応じて、OOMを防止するためのデバイスあたりバッチサイズと勾配累積ステップ数を自動決定
        if vram_limit <= 4.5:
            max_batch = 1
        elif vram_limit <= 8.5:
            max_batch = 4
        else:
            max_batch = 8
        per_device_batch = min(target_total_batch_seqs, max_batch)
        grad_accum_steps = max(1, target_total_batch_seqs // per_device_batch)

    # Pagedオプティマイザが選択されている場合、性能低下の可能性を警告
    optim_selected = config.get("optim", "adamw_torch_fused")
    if "paged" in optim_selected:
        logger.warning(
            f"Paged optimizer '{optim_selected}' is selected. "
            "If GPU VRAM limits are exceeded, optimizer states will be paged to CPU system memory, "
            "which will severely degrade training speed."
        )

    # Hugging Face TrainingArguments の初期化
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=hpo_config.get("max_lr_2d", 3e-4),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,                       # メモリ節約のため勾配チェックポインティングを有効化
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 安定性とコンパイラ互換性のための非再帰方式
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",                        # コサイン学習率スケジューラを採用
        warmup_ratio=hpo_config.get("warmup_ratio", 0.03),
        weight_decay=hpo_config.get("weight_decay", 0.1),
        adam_beta2=hpo_config.get("beta2", 0.95),
        max_grad_norm=hpo_config.get("grad_clip", 1.0),
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 2), # ローカルチェックポイント保持数制限
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.get("eval_steps", 1000) if eval_ds is not None else None,
        logging_steps=config.get("logging_steps", 10),
        report_to=["tensorboard"],                          # TensorBoardによる学習監視の有効化
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,                        # カスタムデータコレーター利用時のカラム自動削除防止
        optim=config.get("optim", "adamw_torch_fused"),
        torch_compile=config.get("torch_compile", False),
        use_liger_kernel=config.get("use_liger_kernel", False),
        dataloader_pin_memory=config.get("dataloader_pin_memory", True),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        torch_empty_cache_steps=config.get("torch_empty_cache_steps", 100),
        dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 2),
    )

    # 11. コールバックの設定
    callbacks = [
        ProgressBarFormatCallback(),                       # 進捗バーの表示をカスタムフォーマット化
        HashSaveCallback(config_hash=current_config_hash, data_hash=current_data_hash),  # hashes.jsonの自動作成
        DetailedLoggingCallback(log_every_n_steps=config.get("logging_steps", 10)),       # 詳細なステップ別メトリクスのログ出力
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # 12. Trainerインスタンスの生成
    from transformers import default_data_collator
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator if config.get("packing", False) else DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    # 詳細ログ出力コールバックにTrainerオブジェクトの参照を渡す
    for cb in callbacks:
        if isinstance(cb, DetailedLoggingCallback):
            cb.trainer = trainer

    # 13. 学習プロセスの実行
    logger.info("*** Starting Unified Training Pipeline ***")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 通常学習の場合、最終結果メトリクスをJSONファイルに保存
    if max_steps == -1:
        with open("last_run_result.json", "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f)

    # 14. 学習済みモデル・トークナイザーの保存処理
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return train_result.training_loss
