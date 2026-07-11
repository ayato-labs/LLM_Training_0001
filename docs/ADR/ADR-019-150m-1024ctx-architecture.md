# ADR-019: 150M / 1024ctx アーキテクチャの確定

- **Status:** Accepted (Updated)
- **Date:** 2026-07-05 (Updated: 2026-07-11)
- **Deciders:** Novel LLM Team

## Context

VRAM 4GB (RTX 3050 Laptop) の制約下で、小説生成タスクに最適なモデル構成を検討。
当初は「100M / 2048ctx」を採用していたが、以下の理由で「150M / 1024ctx」に変更。

### 変更経緯

1. **GQA (Grouped Query Attention) の導入** (ADR-020)
   - KVキャッシュを削減し、150Mでも1024ctxが収まるように
2. **VRAM使用量の再評価**
   - 150M / 1024ctx / ZeRO-1 → 約2.1GB (4GBに余裕)
   - 100M / 2048ctx / ZeRO-3 → 約1.8GB (同等)
3. **表現力の優先**
   - 150Mは100Mの1.5倍のパラメータ → 表現力が向上
   - seq_len=1024は日本語小説で実用十分

### 比較結果 (更新)

| 観点 | 150M / 1024ctx (新) | 100M / 2048ctx (旧) |
|------|---------------------|---------------------|
| パラメータ数 | 150M | 100M |
| シーケンス長 | 1024 | 2048 |
| VRAM使用量 | ~2.1GB | ~1.8GB |
| GQA | 4倍 (12/3) | なし |
| 表現力 | 高い | 低い |
| 文脈保持 | 1章分 | 2章分 |

## Decision

**150M / 1024ctx** を採用する。

### アーキテクチャ詳細

```yaml
model:
  target_params: 150_000_000
  llama:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    num_key_value_heads: 3  # GQA: 12/3 = 4倍
    intermediate_size: 3072
    rope_theta: 10000.0
```

### 根拠
1. **GQAによる効率化**: KVキャッシュを4分の1に削減 → 150Mでも1024ctxが収まる
2. **VRAM余裕**: 2.1GBで4GBに十分余裕 → gradient checkpointing等の最適化が容易
3. **表現力の向上**: 100M→150M で1.5倍のパラメータ → 生成品質の向上が見込める
4. **1024ctxの妥当性**: 日本語小説の平均文長は100-200字 → 1024ctxで1章分をカバー

## Consequences

### メリット
- 100Mの1.5倍の表現力
- GQAによりKVキャッシュを4分の1に削減
- VRAMに余裕 → gradient checkpointing等の最適化が容易
- 学習速度は100M比でやや遅くなるが許容範囲

### デメリット
- seq_len=1024は2048に比べて文脈保持力が低下
- 不過学習リスクがやや増加 (パラメータ数が多いため)

### 次期拡張パス
- Phase 4 (設備投資判断) で 3B/7B へのスケールを検討
- seq_len=2048/4096 への拡張は GPU メモリに依存
