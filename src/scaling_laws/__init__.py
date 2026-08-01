"""Scaling Laws Package (src.scaling_laws)

Hoffmann et al. (Chinchilla), McCandlish/Kaplan (Critical Batch Size),
および Dynamic GPU Proxy Benchmark を責任と関心の分離 (SoC) に従い統合管理。
"""

from src.scaling_laws.calculator import (
    calculate_chinchilla_scaling,
    find_max_safe_batch_size,
)
from src.scaling_laws.chinchilla_law import (
    calculate_compute_optimal_n_d,
    generate_universal_architecture,
    load_base_model_defaults,
)
from src.scaling_laws.critical_batch_law import (
    calculate_critical_batch_size,
    select_decoupled_batch_split,
)
from src.scaling_laws.proxy_benchmark import (
    detect_data_path_and_tokens,
    detect_gpu_info,
    detect_real_available_vram_gb,
    detect_seq_len_from_config,
    extract_throughput_from_recent_logs,
    run_quick_proxy_benchmark,
)

__all__ = [
    "calculate_compute_optimal_n_d",
    "generate_universal_architecture",
    "load_base_model_defaults",
    "calculate_critical_batch_size",
    "select_decoupled_batch_split",
    "detect_data_path_and_tokens",
    "detect_gpu_info",
    "detect_real_available_vram_gb",
    "detect_seq_len_from_config",
    "extract_throughput_from_recent_logs",
    "run_quick_proxy_benchmark",
    "calculate_chinchilla_scaling",
    "find_max_safe_batch_size",
]
