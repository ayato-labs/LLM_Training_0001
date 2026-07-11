# test_bf16.py
import torch
import sys

print("=" * 60)
print("RTX 3050 BF16 対応検証")
print("=" * 60)

# 1. 環境情報
print(f"\n[環境]")
print(f"  Python: {sys.version}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
if torch.cuda.is_available():
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"  FP16 Tensor Core: {torch.cuda.get_device_capability(0)[0] >= 7}")

# 2. BF16 ハードウェアサポート
print(f"\n[BF16 ハードウェアサポート]")
bf16_supported = torch.cuda.is_bf16_supported()
print(f"  torch.cuda.is_bf16_supported(): {bf16_supported}")

# 3. 実行テスト (小さなテンソルで forward/backward)
if bf16_supported and torch.cuda.is_available():
    print(f"\n[実行テスト: BF16 forward + backward]")
    try:
        device = torch.device("cuda")
        x = torch.randn(2, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(512, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

        y = x @ w
        loss = y.sum()

        loss.backward()

        print(f"  OK: BF16 forward/backward 成功")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  x.grad dtype: {x.grad.dtype}")
        print(f"  w.grad dtype: {w.grad.dtype}")
    except Exception as e:
        print(f"  FAIL: エラー: {e}")
        import traceback
        traceback.print_exc()

    # 4. DeepSpeed BF16 互換性チェック
    print(f"\n[DeepSpeed BF16 互換性]")
    try:
        import deepspeed
        print(f"  DeepSpeed version: {deepspeed.__version__}")
        print(f"  OK: DeepSpeed インストール済み")
    except ImportError:
        print(f"  WARN: DeepSpeed 未インストール (pip install deepspeed)")

    # 5. TransformerEngine / FlashAttention チェック
    print(f"\n[高速化ライブラリ]")
    try:
        import flash_attn
        print(f"  FlashAttention: v{flash_attn.__version__}")
    except ImportError:
        print(f"  FlashAttention: 未インストール")
    try:
        import transformer_engine
        print(f"  TransformerEngine: v{transformer_engine.__version__}")
    except ImportError:
        print(f"  TransformerEngine: 未インストール")

else:
    print(f"\n  FAIL: BF16 未サポートまたは CUDA 不可")
    print(f"  bf16_supported={bf16_supported}, cuda={torch.cuda.is_available()}")

print("\n" + "=" * 60)
