# ADR-0032: FlashAttention-2 → SDPA (Scaled Dot Product Attention) への移行

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Solo Developer

## Context

`create_model_config()` で `attn_implementation` のデフォルトが `flash_attention_2` に設定されていたが、以下の問題が発生：

1. **Windows非対応**: FlashAttention-2 はLinux CUDA環境専用。Windows環境でインストール不可。
2. **HPO全試行失敗**: プロキシ学習時に `FlashAttention2 has been toggled on, but it cannot be used` エラーで全試行が 1e9 を返却。
3. **保守コスト**: FlashAttention-2 は追加のビルド手順・CUDAバージョン依存が発生。

## Decision

**PyTorch 2.0+ 標準の Scaled Dot Product Attention (SDPA) に統一** する。

### 変更内容

| ファイル | 変更前 | 変更後 |
|---------|--------|--------|
| `model_utils.py` | `mp.get("attn_implementation", "flash_attention_2")` | `mp.get("attn_implementation", "sdpa")` |
| `config.yaml` | (未設定) | `model.llama.attn_implementation: "sdpa"` |
| `config.py` | (未抽出) | `llama.get("attn_implementation", "sdpa")` 追加 |

### SDPAの利点

- **プラットフォーム互換**: Windows/Linux 両方で動作
- **追加依存不要**: PyTorch 2.0+ に標準搭載
- **性能同等**: FlashAttention-2 と同等の $O(N)$ メモリ効率を達成
- **自動選択**: PyTorch内部で `memory_efficient_attention` → `flash_attention` → `math` を自動選択

## Consequences

### Pros
- **HPO安定稼働**: Windows環境でも全試行が正常実行される
- **依存削減**: `flash-attn` パッケージ不要、ビルド手順不要
- **保守性向上**: CUDAバージョン変更時の追従コスト削減

### Cons
- FlashAttention-2 独自の最適化（一部ハードウェア固有）は利用不可
- ただし SDPA も PyTorch 内部で同等最適化を自動適用するため、実質的な性能差は最小

## 参照

- ADR-0026: Scratch Pre-training Optimizations
- 概念的要件定義書: §2.1 採用する標準スタック
