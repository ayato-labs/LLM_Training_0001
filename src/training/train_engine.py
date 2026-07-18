import os
import json
from pathlib import Path

import torch
import torch._dynamo
from datasets import disable_caching, load_dataset

# Linuxの /tmp (tmpfs RAMディスク) の容量不足による [Errno 28] OOM を回避するため、
# 一時ディレクトリを十分な空き容量のあるローカルディスク（ext4）上に強制指定
os.environ["TMPDIR"] = str(Path("models/output/tmp").resolve())
Path("models/output/tmp").mkdir(parents=True, exist_ok=True)

disable_caching()

from transformers import (
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.common.logger import logger
from src.common.set_seed import set_seed
from src.training.callbacks import (
    DetailedLoggingCallback,
    HashSaveCallback,
)
from src.training.model_utils import (
    PackedDatasetWrapper,
    compute_dataset_fingerprint,
    compute_db_fingerprint,
    compute_file_hash,
    create_model_config,
    get_optimal_num_proc,
    parallel_tokenize,
)


def _verify_flash_attention(precision: str) -> None:
    """SDPA経由でFlashAttentionバックエンドが正常にディスパッチ可能かを明示的に検証する（サイレントフォールバック防止）。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Skipping FlashAttention verification.")
        return

    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    try:
        # Llamaモデルの標準ヘッド構造を模したテンソルを作成
        q = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")
        k = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")
        v = torch.randn(1, 12, 1024, 64, dtype=dtype, device="cuda")

        # FLASH_ATTENTIONのみを有効化した状態で実行テスト
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        logger.info(f"Verification SUCCESS: Native FlashAttention backend is active and verified via SDPA (dtype={precision}).")
    except Exception as e:
        logger.warning(
            f"Verification WARNING: FlashAttention backend could not be dispatched via SDPA. "
            f"PyTorch will silently fall back to slower math kernels. Error: {e}"
        )


def train(config: dict, tokenized_datasets=None, extra_callbacks=None):
    """
    統一された学習オーケストレーションフロー。

    Args:
        config (dict): 学習設定パラメータを含む辞書。
        tokenized_datasets (dict, optional): トークン化済みのデータセット辞書（学習/検証）。省略時はロード及びトークン化を行う。
        extra_callbacks (list, optional): 追加のTrainerCallbackリスト。
    """
    # 0. 解決されたハイパーパラメータ/設定値の全出力
    logger.info(f"Resolved Configuration:\n{json.dumps(config, indent=2, ensure_ascii=False)}")

    # 1. 乱数シードの設定（再現性確保のため決定論的挙動を強制）
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True)

    # 1.5 TF32 (TensorFloat-32) の有効化（Ampere世代以降のGPUでの行列演算高速化）
    if config.get("allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for matmul and cudnn.")

    # 1.6 torch.compile コンパイラ最適化設定 (Graph break の抑制)
    torch._dynamo.config.capture_scalar_outputs = True

    # 1.7 FlashAttention のディスパッチ検証（サイレントフォールバックの防止）
    _verify_flash_attention(config.get("precision", "bf16"))

    # 2. 現在の設定ファイルとデータセットのハッシュ値の算出（整合性チェック用）
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
    if resume_checkpoint is True or (
        isinstance(resume_checkpoint, str) and "checkpoint-latest" in resume_checkpoint
    ):
        from src.training.model_utils import get_checkpoints

        checkpoints = get_checkpoints(sort_by="mtime")
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
                with open(hash_file) as f:
                    saved_hashes = json.load(f)
                saved_config_hash = saved_hashes.get("config_hash")
                saved_data_hash = saved_hashes.get("data_hash")

                # チェックポイント保存時と現在の設定・データが異なる場合はエラーとして学習を中断する
                if saved_config_hash != current_config_hash or saved_data_hash != current_data_hash:
                    logger.error(
                        "Configuration or training dataset has changed since the checkpoint was saved!"
                    )
                    logger.error(
                        f"Saved Config Hash: {saved_config_hash} | Current Config Hash: {current_config_hash}"
                    )
                    logger.error(
                        f"Saved Data Hash: {saved_data_hash} | Current Data Hash: {current_data_hash}"
                    )
                    raise ValueError(
                        "Cannot resume training: config.yaml or training dataset does not match the checkpoint."
                    )
                else:
                    logger.info("Configuration and dataset hashes match. Verification successful.")
            except Exception as e:
                logger.error(f"Failed to verify checkpoint hashes: {e}")
                raise
        else:
            logger.warning(
                f"No hashes.json found in checkpoint {resume_checkpoint}. Proceeding without verification."
            )
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

        # プラットフォーム自動判定で並列トークナイズ (Windows: ThreadPool, Linux: multiprocessing)
        num_proc = get_optimal_num_proc()
        logger.info(f"Tokenizing dataset with parallelism={num_proc}")

        ds = parallel_tokenize(
            ds,
            tokenizer,
            seq_len=seq_len,
            padding=not packing,
            remove_columns=remove_columns,
            max_workers=num_proc,
            batch_size=config.get("tokenization", {}).get("batch_size", 1000),
        )

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

    # Pagedオプティマイザが選択されている場合、性能低下の可能性を警告
    optim_selected = config.get("optim", "adamw_torch_fused")
    if "paged" in optim_selected:
        logger.warning(
            f"Paged optimizer '{optim_selected}' is selected. "
            "If GPU VRAM limits are exceeded, optimizer states will be paged to CPU system memory, "
            "which will severely degrade training speed."
        )

    # Hugging Face TrainingArguments の初期化
    # warmup_steps と warmup_ratio の動的設定 (epochモード max_steps=-1 時の比率解決)
    warmup_steps = hpo_config.get("warmup_steps", 0)
    warmup_ratio = hpo_config.get("warmup_ratio", 0.03)
    if warmup_steps > 0:
        warmup_ratio = 0.0
    else:
        warmup_steps = None

    # dataloader_num_workers の動的設定 (Linux/WSLの場合、os.cpu_count()を用いてデータ準備を非同期化)
    num_workers = config.get("dataloader_num_workers", 0)
    if num_workers == 0:
        import os
        import sys
        if sys.platform == "linux":
            num_workers = min(4, max(1, (os.cpu_count() or 2) // 4))
            logger.info(f"Auto-selected dataloader_num_workers: {num_workers} (system CPU count: {os.cpu_count()})")

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=hpo_config.get("max_lr_2d", 3e-4),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,  # メモリ節約のため勾配チェックポインティングを有効化
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # 安定性とコンパイラ互換性のための非再帰方式
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",  # コサイン学習率スケジューラを採用
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=hpo_config.get("weight_decay", 0.1),
        adam_beta2=hpo_config.get("beta2", 0.95),
        max_grad_norm=hpo_config.get("grad_clip", 1.0),
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        save_strategy="steps",
        save_steps=config.get("save_steps", 1000),
        save_total_limit=config.get("save_total_limit", 2),  # ローカルチェックポイント保持数制限
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.get("eval_steps", 1000) if eval_ds is not None else None,
        logging_steps=config.get("logging_steps", 10),
        report_to=["tensorboard"],  # TensorBoardによる学習監視の有効化
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,  # カスタムデータコレーター利用時のカラム自動削除防止
        optim=config.get("optim", "adamw_torch_fused"),
        torch_compile=config.get("torch_compile", False),
        use_liger_kernel=config.get("use_liger_kernel", False),
        dataloader_pin_memory=config.get("dataloader_pin_memory", True),
        dataloader_num_workers=num_workers,
        torch_empty_cache_steps=config.get("torch_empty_cache_steps", 100),
        dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 2)
        if num_workers > 0
        else None,
        disable_tqdm=True,  # tqdm進捗バー無効化（独自ログのみ使用）
    )

    # 11. コールバックの設定
    callbacks = [
        HashSaveCallback(
            config_hash=current_config_hash, data_hash=current_data_hash
        ),  # hashes.jsonの自動作成
        DetailedLoggingCallback(
            log_every_n_steps=config.get("logging_steps", 10)
        ),  # 詳細なステップ別メトリクスのログ出力
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
        data_collator=default_data_collator
        if config.get("packing", False)
        else DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    # 既定のPrinterCallback（disable_tqdm=True時に標準出力へ辞書をprintする）を削除して重複ログを防止
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)

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
