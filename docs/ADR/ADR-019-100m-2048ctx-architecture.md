# ADR-019: 100M / 2048ctx アーキテクチャの確定

- **Status:** Accepted
- **Date:** 2026-07-05
- **Deciders:** Novel LLM Team

## Context

VRAM 4GB (RTX 3050 Laptop) の制約下で、小説生成タスクに最適なモデル構成を検討。
「150M / 512ctx」と「100M / 2048ctx」のトレードオフを比較。

### 比較結果

| 観点 | 150M / 512ctx | 100M / 2048ctx |
|------|--------------|----------------|
| 小説の文脈保持 | 1-2段落で切れる | 1章(1000字以上)収まる |
| メタデータ制御 | metadata+本文で溢れる | metadata+長文本文余裕 |
| VRAM (ZeRO-3) | ~2.5GB (ギリギリ) | ~1.8GB (余裕) |
| 学習速度 | 基準 | ~1.3倍高速 (params少) |
| 実用品質(小説) | 文脈切れで崩壊 | 長距離依存学習可能 |

## Decision

**100M / 2048ctx** を採用する。

### アーキテクチャ詳細

```yaml
model:
  target_params: 100000000
  llama:
    hidden_size: 512
    num_hidden_layers: 16
    num_attention_heads: 8
    intermediate_size: 2048
    rope_theta: 50000.0
```

### 根拠
1. **小説生成は文脈長が支配的**: 512ctx では 1-2段落で切れるため、章単位の構成・伏線回収・キャラ一貫性が学習できない
2. **VRAM 余裕**: 100M/2048ctx なら ZeRO-3 で ~1.8GB。4GB に十分余裕
3. **スケーリング則**: データ不足下 (30M tokens) では、N を減らして D を増やす方が損失は下がる
4. **PoC としての判断**: 100M の結果を見て、設備投資を判断する

## Consequences

### メリット
- 小説1章相当の文脈を保持可能
- VRAM に余裕 → 他の最適化 (packed sequence, gradient checkpointing) が容易
- 学習速度が高速

### デメリット
- 表現力が 150M に比べて低下 (PoC としては許容範囲)

### 次期拡張パス
- Phase 4 (設備投資判断) で 300M/1B/3B へのスケールを検討
