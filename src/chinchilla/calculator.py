"""Universal Chinchilla Calculator Core

特定の GPU 型番や固定テーブルに依存せず、任意の GPU 仕様 (4GB ~ 80GB VRAM)、
直近ログ/プロキシベンチマークからの自動実測スループット、および幾何学的モデル生成数式を用いて
チンチラ最適 (Chinchilla Optimal) なモデルパラメータ数とモデル構造を完全自律算定する。
"""

import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.common.logger import logger


def detect_data_path_and_tokens() -> tuple[str, float | None]:
    """configs/base_config.yaml から data_path と tokenizer_path を取得し、
    本番と同一の PreTrainedTokenizerFast をファンクションコールして正確な総トークン数を算定
    """
    config_path = Path("configs/base_config.yaml")
    data_path = "data/dataset.jsonl"
    tokenizer_path = "data/tokenizer.json"

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                if "data_path" in cfg and cfg["data_path"]:
                    data_path = str(cfg["data_path"])
                if "tokenizer_path" in cfg and cfg["tokenizer_path"]:
                    tokenizer_path = str(cfg["tokenizer_path"])
        except Exception:
            pass

    path_obj = Path(data_path)
    if not path_obj.exists():
        return data_path, None

    try:
        from transformers import PreTrainedTokenizerFast

        # 本番と同一の PreTrainedTokenizerFast をファンクションコールしてインスタンス化
        tokenizer = None
        if Path(tokenizer_path).exists():
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        file_size_bytes = path_obj.stat().st_size

        if tokenizer is not None:
            # データセット先頭からサンプル行を抽出し、正確な 1 バイトあたりのトークン密度 (tokens_per_byte) を計算
            sample_text = ""
            sample_bytes = 0
            with open(path_obj, encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    sample_text += line
                    sample_bytes += len(line.encode("utf-8"))
                    if idx >= 200 or sample_bytes >= 200_000:
                        break

            if sample_bytes > 0 and sample_text:
                sample_tokens = len(tokenizer.encode(sample_text))
                tokens_per_byte = sample_tokens / sample_bytes
                total_tokens = file_size_bytes * tokens_per_byte
                return data_path, float(total_tokens)

        # トークナイザー未ロード時の物理バックアップ
        est_tokens = file_size_bytes / 3.5
        return data_path, float(est_tokens)
    except Exception as e:
        logger.warning(f"Could not estimate dataset tokens via PreTrainedTokenizerFast from {data_path}: {e}")
        return data_path, None



def detect_seq_len_from_config() -> int:
    """configs/base_config.yaml から事前学習の基本 seq_len を一元取得"""
    config_path = Path("configs/base_config.yaml")
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and "training" in cfg and "seq_len" in cfg["training"]:
                return int(cfg["training"]["seq_len"])
        except Exception as e:
            logger.warning(f"Failed to read seq_len from base_config.yaml: {e}")
    return 1024


def detect_gpu_info() -> dict[str, Any]:
    """GPUの物理仕様および演算能力を動的に自動取得"""
    if not torch.cuda.is_available():
        return {
            "device_name": "CPU (CUDA Unavailable)",
            "total_vram_gb": 4.0,
            "sm_count": 0,
            "is_cuda": False,
        }

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    sm_count = getattr(props, "multi_processor_count", 0)

    return {
        "device_name": props.name,
        "total_vram_gb": round(vram_gb, 2),
        "sm_count": sm_count,
        "is_cuda": True,
    }


def extract_throughput_from_recent_logs() -> float | None:
    """最新のログファイルから直近の訓練スループット (tokens/sec) を自動抽出"""
    # ログファイルの候補探索
    log_paths = list(Path(".").glob("**/logs/*.log")) + list(Path(".").glob("*.log"))
    if not log_paths:
        return None

    # 最新の修正日時を持つログファイルを選択
    log_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    pattern = re.compile(r"(\d+\.\d+)s/it")
    for log_path in log_paths[:3]:
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                match = pattern.search(line)
                if match:
                    sec_per_it = float(match.group(1))
                    if sec_per_it > 0:
                        # 1 iteration = 1 step = seq_len(1024) * grad_accum(32) = 32,768 tokens
                        tokens_per_it = 1024 * 32
                        tps = tokens_per_it / sec_per_it
                        return round(tps, 1)
        except Exception:
            continue
    return None


def run_quick_proxy_benchmark(seq_len: int = 1024, batch_size: int = 16) -> tuple[float, int]:
    """指定の seq_len および batch_size で GPU 上のプロキシモデルを実際に動的実行し、
    実効スループット (tokens/sec) および 実測パラメータ数を完全リアルタイム計測。
    """
    if not torch.cuda.is_available():
        return 500.0, 27_450_000

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    target_bs = max(1, batch_size)

    while target_bs >= 1:
        try:
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaModel

            cfg = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                intermediate_size=1376,
            )
            device = torch.device("cuda:0")
            model = LlamaModel(cfg).to(device=device, dtype=torch.bfloat16)
            ref_n = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Warmup step (指定 seq_len と target_bs でデモ実行)
            dummy_input = torch.randint(0, 32000, (target_bs, seq_len), device=device)
            out = model(dummy_input)
            loss = out.last_hidden_state.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()

            # Benchmark 3 steps
            start_t = time.time()
            n_steps = 3
            for _ in range(n_steps):
                out = model(dummy_input)
                loss = out.last_hidden_state.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            elapsed = time.time() - start_t
            sec_per_step = elapsed / n_steps
            tps = (target_bs * seq_len) / sec_per_step

            del model, optimizer
            torch.cuda.empty_cache()
            return round(tps, 1), ref_n
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if target_bs <= 1:
                raise
            target_bs = target_bs // 2
            logger.info(f"run_quick_proxy_benchmark OOM, retrying with batch_size={target_bs}...")
        except Exception as e:
            logger.error(f"run_quick_proxy_benchmark failed for seq_len={seq_len}, batch_size={target_bs}: {e}")
            raise



def estimate_peak_vram_gb(
    n_params: int,
    seq_len: int = 1024,
    batch_size: int = 16,
    selective_checkpointing: bool = True,
    vram_cap: float = 4.0,
    hidden_size: int = 768,
    num_layers: int = 12,
    vocab_size: int = 32000,
    intermediate_size: int = 0,
    precision: str = "bf16",
    optimizer_type: str = "adamw_bnb_8bit",
    use_liger_kernel: bool = True,
    torch_compile: bool = False,
) -> float:
    """統一VRAM推定エンジン (src.common.vram_estimator) に委譲してピークVRAM (GB) を返す。

    既存の呼び出し元との互換性のため、引数シグネチャは維持しつつ、
    詳細見積もりパラメータ（intermediate_size, precision, optimizer_type等）をオプションで追加。
    """
    from src.common.vram_estimator import VramConfig, estimate_training_vram

    # selective_checkpointing → checkpointing モード変換
    checkpointing = "selective" if selective_checkpointing else "full"

    # intermediate_size=0 の場合は auto (4 * hidden_size) 扱い
    eff_intermediate = intermediate_size if intermediate_size > 0 else 4 * hidden_size

    cfg = VramConfig(
        n_params=n_params,
        hidden_size=hidden_size,
        intermediate_size=eff_intermediate,
        num_layers=num_layers,
        vocab_size=vocab_size,
        micro_batch_size=batch_size,
        seq_len=seq_len,
        precision=precision,  # type: ignore[arg-type]
        optimizer_type=optimizer_type,  # type: ignore[arg-type]
        checkpointing=checkpointing,  # type: ignore[arg-type]
        use_liger_kernel=use_liger_kernel,
        torch_compile=torch_compile,
        total_vram_gb=vram_cap,
    )

    est = estimate_training_vram(cfg)
    return round(est.breakdown.total_estimated_gb, 2)


def load_base_model_defaults() -> dict[str, Any]:
    """モデルアーキテクチャのデフォルト標準パラメータ（単一のSSOTフォールバック）"""
    return {
        "vocab_size": 32000,
        "rope_theta": 10000.0,
        "attn_implementation": "sdpa",
        "tie_word_embeddings": True,
        "gqa_ratio": 4,
        "head_dim_default": 64,
        "head_dim_large": 128,
        "head_dim_large_threshold_params": 3_000_000_000,
    }


def generate_universal_architecture(target_n_params: int) -> dict[str, Any]:
    """本番共通モジュール (src.training.model_utils) をファンクションコールして
    任意のターゲットパラメータ数 N (10M ~ 70B) に対する LlamaConfig を生成し、
    本番と 100% 一致するモデルアーキテクチャを決定。
    """
    model_defs = load_base_model_defaults()

    large_threshold = model_defs.get("head_dim_large_threshold_params", 3_000_000_000)
    head_dim_large = model_defs.get("head_dim_large", 128)
    head_dim_default = model_defs.get("head_dim_default", 64)
    head_dim = head_dim_large if target_n_params >= large_threshold else head_dim_default

    # 概算のアスペクト比から Attention Head 数を逆算
    # Llama/Transformers の制約: n_heads % kv_heads == 0 (完全な整数倍)
    est_h = (target_n_params / 0.216) ** (1 / 3)
    gqa_ratio = model_defs.get("gqa_ratio", 4)
    raw_heads = max(4, round(est_h / head_dim))

    # 1. target kv_heads を決定 (最小1)
    kv_heads = max(1, round(raw_heads / gqa_ratio))
    # 2. n_heads を kv_heads の倍数にアライン (最小4)
    multiplier = max(1, round(raw_heads / kv_heads))
    n_heads = max(4, multiplier * kv_heads)

    hidden_size = n_heads * head_dim

    # SwiGLU FFN 次元算定 (8/3 * H を 128 の倍数にアラインメント)
    raw_ffn = int(hidden_size * 8 / 3)
    intermediate_size = max(512, (raw_ffn // 128) * 128)

    # 重み共有 (Weight Tying) および語彙数を考慮した層数 L の逆算
    params_per_layer = (4 * hidden_size * hidden_size) + (3 * hidden_size * intermediate_size)
    vocab_size = model_defs.get("vocab_size", 32000)
    embed_params = vocab_size * hidden_size

    remaining_params = max(0, target_n_params - embed_params)
    n_layers = max(4, round(remaining_params / params_per_layer))

    arch_dict = {
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": kv_heads,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "rope_theta": model_defs.get("rope_theta", 10000.0),
        "attn_implementation": model_defs.get("attn_implementation", "sdpa"),
        "tie_word_embeddings": model_defs.get("tie_word_embeddings", True),
    }

    # 本番学習で使われる src.training.model_utils.create_model_config をファンクションコール
    exact_n_params = embed_params + (n_layers * params_per_layer)
    try:
        from unittest.mock import MagicMock

        from transformers import LlamaForCausalLM

        from src.training.model_utils import create_model_config, estimate_model_size

        dummy_tokenizer = MagicMock()
        dummy_tokenizer.pad_token_id = 0
        dummy_tokenizer.bos_token_id = 1
        dummy_tokenizer.eos_token_id = 2

        config_dict = {
            "model_params": arch_dict,
            "seq_len": 1024,
            "attn_implementation": arch_dict["attn_implementation"],
        }
        # 本番と 100% 同一の LlamaConfig を動的生成
        llama_config = create_model_config(config_dict, dummy_tokenizer)

        # 本番と同一の PyTorch パラメータ数カウント
        with torch.device("meta"):
            meta_model = LlamaForCausalLM(llama_config)
            exact_n_params = estimate_model_size(meta_model)
            del meta_model

        arch_dict["hidden_size"] = llama_config.hidden_size
        arch_dict["num_hidden_layers"] = llama_config.num_hidden_layers
    except Exception as e:
        logger.debug(f"create_model_config fallback: {e}")

    arch_dict["n_params"] = exact_n_params
    return arch_dict


from typing import Any

from src.hpo.hpo_manager import calculate_dynamic_n_trials


def detect_real_available_vram_gb(device_id: int = 0) -> float:
    """
    CUDAコンテキストを強制初期化し、現在のシステムにおける
    「純粋にモデル学習に使用可能な真の空きVRAM」を動的計測する。
    """
    if not torch.cuda.is_available():
        return 4.0  # CPU Fallback

    try:
        # 1. 強制的にCUDAコンテキストを発火
        # これによりOSのWDDMやPyTorchのベースフットプリントがVRAMに確保される
        dummy = torch.zeros(1, device=f"cuda:{device_id}")
        del dummy

        # 2. キャッシュをクリアし、純粋な空き状態を作る
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 3. OS/ドライバレベルでの空きメモリを取得 (Bytes)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)

        # 4. GB単位に変換
        free_vram_gb = free_bytes / (1024**3)
        return round(free_vram_gb, 2)
    except Exception as e:
        logger.warning(f"Failed to detect real VRAM, falling back to total VRAM estimation: {e}")
        props = torch.cuda.get_device_properties(device_id)
        return round((props.total_memory / (1024**3)) - 0.8, 2)  # 0.8GBを静的マージンとする


def _vram_estimate(
    n_params: int, seq_len: int, batch_size: int,
    hidden_size: int, num_layers: int, vocab_size: int,
    intermediate_size: int = 0,
    checkpointing: str = "selective",
) -> "VramEstimate":
    from src.common.vram_estimator import (
        VramConfig,
        VramEstimate,
        estimate_training_vram_with_calibration,
    )

    return estimate_training_vram_with_calibration(VramConfig(
        n_params=n_params, hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_layers=num_layers, vocab_size=vocab_size,
        micro_batch_size=batch_size, seq_len=seq_len,
        precision="bf16", optimizer_type="adamw_bnb_8bit",
        use_liger_kernel=True, checkpointing=checkpointing,
        total_vram_gb=999.0,
    ))


def find_max_safe_batch_size(
    arch_dict: dict,
    seq_len: int,
    true_free_vram_gb: float,
    max_search_bs: int = 64,
    checkpointing: str = "selective",
) -> int:
    """
    指定されたアーキテクチャと空きVRAMにおいて、OMMを起こさない最大のバッチサイズを自動探索する。
    """
    safe_limit_gb = true_free_vram_gb * 0.90

    optimal_bs = 1
    for test_bs in range(max_search_bs, 0, -1):
        est_vram = _vram_estimate(
            arch_dict["n_params"], seq_len, test_bs,
            arch_dict["hidden_size"], arch_dict["num_hidden_layers"],
            arch_dict["vocab_size"], arch_dict.get("intermediate_size", 0),
            checkpointing=checkpointing,
        ).breakdown.total_estimated_gb
        if est_vram <= safe_limit_gb:
            optimal_bs = test_bs
            break

    return optimal_bs


def calculate_chinchilla_scaling(
    target_hours: float = 48.0,
    user_throughput_tps: float | None = None,
    user_seq_len: int | None = None,
    user_vram_limit_gb: float | None = None,
    force_benchmark: bool = False,
    reference_n_params: int | None = None,
) -> dict[str, Any]:
    """目標時間、実効FLOPs/sec、および動的VRAM検知から完全自律型の最適構成を算定
    
    Args:
        target_hours: 目標学習時間 (hours)
        user_throughput_tps: ユーザー指定スループット (tokens/sec)
        user_seq_len: ユーザー指定シーケンス長
        user_vram_limit_gb: ユーザー指定VRAM上限 (GB)
        force_benchmark: 強制的にGPUベンチマーク実行
        reference_n_params: スループット基準モデルのパラメータ数
    """

    gpu_info = detect_gpu_info()
    seq_len = user_seq_len or detect_seq_len_from_config()

    # --- 1. 動的VRAM検知 (Dynamic Provisioning) ---
    if user_vram_limit_gb:
        true_free_vram_gb = user_vram_limit_gb
        logger.info(f"Using user-specified VRAM limit: {true_free_vram_gb} GB")
    else:
        true_free_vram_gb = detect_real_available_vram_gb()
        logger.info(f"Detected true available VRAM (after CUDA init): {true_free_vram_gb} GB")

    # --- 2. TPSと実効FLOPsの算定 ---
    tp_source = "User Specified"
    tps = user_throughput_tps
    ref_n = reference_n_params

    if tps is None and not force_benchmark:
        extracted = extract_throughput_from_recent_logs()
        if extracted:
            tps = extracted
            tp_source = "Extracted from Recent Training Logs"
            if ref_n is None:
                ref_n = 1_000_000_000
                logger.warning("Assuming 1B params for log TPS. Use force_benchmark=True for precision.")

    if tps is not None and ref_n is None:
        ref_n = 1_000_000_000

    if tps is None or force_benchmark:
        proxy_n = 27_450_000
        proxy_arch_dict = generate_universal_architecture(proxy_n)
        bench_bs = find_max_safe_batch_size(proxy_arch_dict, seq_len, true_free_vram_gb, checkpointing="selective")
        logger.info(f"Running GPU proxy benchmark for seq_len={seq_len}, batch_size={bench_bs}...")
        tps, ref_n = run_quick_proxy_benchmark(seq_len=seq_len, batch_size=bench_bs)
        tp_source = f"Dynamic GPU Benchmark (seq_len={seq_len}, batch={bench_bs})"

    # C = 6 * N * TPS
    effective_flops_per_sec = 6.0 * ref_n * tps
    total_seconds = target_hours * 3600.0
    total_flops_computable = effective_flops_per_sec * total_seconds

    # Chinchilla Optimal: N = sqrt(C / 120), D = 20 * N
    chinchilla_n_pure = math.sqrt(total_flops_computable / 120.0)
    total_tokens_computable = 20.0 * chinchilla_n_pure

    # --- 3. データセット限界の検知 ---
    data_path, actual_dataset_tokens = detect_data_path_and_tokens()
    data_shortage_warn = False
    data_capped_arch = None
    data_sufficiency_ratio = 100.0

    max_allowed_n_by_data = float('inf')

    if actual_dataset_tokens is not None:
        data_sufficiency_ratio = round((actual_dataset_tokens / total_tokens_computable) * 100.0, 1)
        if total_tokens_computable > actual_dataset_tokens:
            data_shortage_warn = True
            max_allowed_n_by_data = max(20_000_000, actual_dataset_tokens / 20.0)
            data_capped_arch = generate_universal_architecture(int(max_allowed_n_by_data))

    # --- 4. VRAM上限 N の二分探索 (バッチサイズ >= 2 が担保できる最大サイズ) ---
    def is_n_vram_safe(test_n: int) -> bool:
        test_arch = generate_universal_architecture(test_n)
        # 最低限のバッチサイズ(=2)で安全マージン内に収まるか判定
        bs = find_max_safe_batch_size(test_arch, seq_len, true_free_vram_gb, checkpointing="selective")
        return bs >= 2

    low_n, high_n = 20_000_000, 2_000_000_000
    max_vram_safe_n = low_n
    while low_n <= high_n:
        mid_n = (low_n + high_n) // 2
        if is_n_vram_safe(mid_n):
            max_vram_safe_n = mid_n
            low_n = mid_n + 100_000
        else:
            high_n = mid_n - 100_000
        if high_n - low_n < 100_000:
            break

    # 計算予算、データ上限、VRAM物理限界の3つから最終的なパラメータ数を決定
    target_n = min(chinchilla_n_pure, max_vram_safe_n, max_allowed_n_by_data)
    target_n = max(target_n, 20_000_000)

    # --- 5. 最適アーキテクチャの生成とバッチサイズの自動決定 ---
    arch = generate_universal_architecture(int(target_n))

    # 決定されたアーキテクチャにおける「最大安全バッチサイズ」を算出
    optimal_batch_size = find_max_safe_batch_size(arch, seq_len, true_free_vram_gb, checkpointing="selective")

    # 最終的なVRAM見積もり (ロギング用)
    est_vram = _vram_estimate(
        arch["n_params"], seq_len, optimal_batch_size,
        arch["hidden_size"], arch["num_hidden_layers"],
        arch["vocab_size"], arch.get("intermediate_size", 0),
    ).breakdown.total_estimated_gb
    vram_safe = est_vram <= (true_free_vram_gb * 0.90)

    # --- 6. 理論的計算量と速度の予測 ---
    # Chinchilla最適トークン数 D（不変・batch_size非依存）
    computable_tokens = total_tokens_computable

    # FLOPsからターゲットモデルのTPSを予測
    projected_target_tps = effective_flops_per_sec / (6.0 * arch["n_params"])

    # --- 7. HPO (プロキシ) 用のシミュレーションとバッチサイズ調整 ---
    proxy_params = arch["n_params"] if arch["n_params"] <= 50_000_000 else min(arch["n_params"], max(50_000_000, int(arch["n_params"] * 0.10)))
    try:
        hpo_n_trials = calculate_dynamic_n_trials(search_space_dim=5)
        proxy_arch_dict = generate_universal_architecture(proxy_params)
        proxy_params = proxy_arch_dict["n_params"]
    except Exception:
        hpo_n_trials = 150
        proxy_arch_dict = generate_universal_architecture(proxy_params)

    # プロキシモデルに対してもOMMを防ぐ安全なバッチサイズを自動計算
    hpo_optimal_batch_size = find_max_safe_batch_size(proxy_arch_dict, seq_len, true_free_vram_gb, checkpointing="selective")
    hpo_trial_steps = 100

    proxy_trial_tokens = seq_len * hpo_optimal_batch_size * hpo_trial_steps
    projected_proxy_tps = effective_flops_per_sec / (6.0 * proxy_params)
    proxy_sec_per_trial = proxy_trial_tokens / projected_proxy_tps

    hpo_worst_case_sec = proxy_sec_per_trial * hpo_n_trials
    hpo_worst_case_min = round(hpo_worst_case_sec / 60.0, 1)
    hpo_expected_min = round(hpo_worst_case_min * 0.35, 1)

    # main.py との互換性を維持したフラットな返り値構造
    return {
        "gpu_info": gpu_info,
        "target_hours": target_hours,
        "seq_len": seq_len,

        "measured_throughput_tps": round(tps, 1),
        "throughput_source": tp_source,
        "reference_model_params": ref_n,
        "effective_tflops": round(effective_flops_per_sec / 1e12, 2),

        "computable_tokens": round(total_tokens_computable),
        "computable_tokens_million": round(total_tokens_computable / 1e6, 2),
        "dataset_info": {
            "data_path": data_path,
            "actual_tokens_million": round(actual_dataset_tokens / 1e6, 2) if actual_dataset_tokens else None,
            "sufficiency_ratio": data_sufficiency_ratio,
            "is_data_shortage": data_shortage_warn,
        },
        "chinchilla_pure_optimal_n_million": round(chinchilla_n_pure / 1e6, 2),
        "recommended_architecture": arch,
        "data_capped_architecture": data_capped_arch,

        "true_free_vram_gb": true_free_vram_gb,
        "estimated_peak_vram_gb": est_vram,
        "vram_limit_gb": gpu_info.get("total_vram_gb", 4.0),
        "is_vram_safe": vram_safe,

        # OMM防止のため動的算出された最適なバッチサイズをエクスポート
        "optimal_batch_size": optimal_batch_size,

        "projected_target_tps": round(projected_target_tps, 1),
        "hpo_simulation": {
            "proxy_params": proxy_params,
            "n_trials": hpo_n_trials,
            "hpo_optimal_batch_size": hpo_optimal_batch_size,
            "worst_case_no_pruning_minutes": hpo_worst_case_min,
            "expected_with_median_pruner_minutes": hpo_expected_min,
        },
    }
