# ADR-018: BF16 (BFloat16) 採用による学習安定性向上

- **Status:** Accepted
- **Date:** 2026-07-05
- **Deciders:** Novel LLM Team

## Context

現在の学習パイプラインは DeepSpeed で `fp16: { "enabled": true }` を使用している。
FP16 は動的範囲が ±65,504 と狭く、学習中の勾配アンダーフロー防止のために Loss Scaling が必要。
Loss Scaling は動的係数調整 (loss_scale_window) を含め複雑で、不安定さや NaN 発散リスクの原因となる。

RTX 3050 Laptop (Ampere, Compute Capability 8.6) は BF16 ハードウェアに対応しており、
`torch.cuda.is_bf16_supported()` で検証済み。forward/backward テストも成功。

## Decision

FP16 から BF16 に切り替える。

### 具体的変更

| 項目 | FP16 (旧) | BF16 (新) |
|------|-----------|-----------|
| DeepSpeed設定 | `"fp16": {"enabled": true}` | `"bf16": {"enabled": true}` |
| Loss Scaling | 必須 (動的調整) | 不要 |
| 数値範囲 | ±65,504 | ±3.4×10³⁸ (FP32同等) |
| 尾数精度 | 10bit | 7bit |
| メモリ | 2 bytes/param | 2 bytes/param (変化なし) |

## Consequences

### メリット
- Loss Scaling 不要 → 学習安定性が大幅向上
- NaN/発散リスクが大幅減少
- メモリ・学習速度は変化なし
- RTX 3050 (Ampere) で検証済み

### デメリット
- 尾数精度が 10bit→7bit に低下 (実用上ほぼ問題なし)
- Pascal以前のGPUでは使用不可 (現環境は Ampere なので影響なし)

### 影響範囲
- `train_model.py`: `generate_deepspeed_config()` の bf16 デフォルト化
- `ds_config.json`: bf16 版への更新
- `configs/config.yaml`: `hardware.precision: "bf16"` 追加
