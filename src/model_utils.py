from src.modern_gpt import ModernGPTConfig

def estimate_config_from_params(target_params: int) -> ModernGPTConfig:
    # 目標は 5% または 30M
    params = max(int(target_params * 0.05), 30_000_000)
    
    # Llama-like scaling: N ≈ 12 * L * H^2
    # L=12固定としてHを逆算
    L = 12
    H = int((params / (12 * L)) ** 0.5)
    
    # 安定性のための制約: n_head=12 とし、Hがその倍数になるように調整
    n_head = 12
    H = (H // n_head) * n_head
    
    return ModernGPTConfig(
        n_layer=L,
        n_embd=H,
        n_head=n_head,
        n_kv_head=max(1, n_head // 4),
        vocab_size=32768
    )
