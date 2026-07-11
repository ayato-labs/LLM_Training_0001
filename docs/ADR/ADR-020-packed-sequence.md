# ADR-020: Packed Sequence 導入による学習効率向上

- **Status:** Accepted
- **Date:** 2026-07-05
- **Deciders:** Novel LLM Team

## Context

現在の学習は `padding="max_length"` で全サンプルを固定長 512/2048 にパディングしている。
短文小説では 60-90% が `<PAD>` トークンで、GPU 計算の大部分が無駄になっている。

## Decision

Packed Sequence を導入し、全サンプルを連結して `seq_len` で分割する。

### 変更前 (固定長パディング)
```
[小説A: 200 token] + [PAD × 1848]  → 2048 token (90% が PAD)
[小説B: 450 token] + [PAD × 1598]  → 2048 token (78% が PAD)
```

### 変更後 (Packed Sequence)
```
[A: 200][EOS][B: 450][EOS][C: 50][EOS]...  → 512/2048 で分割
余りは次バッチへ
```

### 実装方針
- HuggingFace `datasets` の `map()` で `group_texts()` を適用
- `DataCollatorForLanguageModeling(mlm=False)` はそのまま使用
- `attention_mask` は全1 (パディングなし)

## Consequences

### メリット
- パディングほぼゼロ → 実効バッチサイズ 2-3倍
- 同ステップ数で 2-3倍のトークン処理
- 文脈跨ぎ学習 → 章境界・作品境界も自然に学習

### デメリット
- 実装工数: ~2時間
- 文境界 (`<|end_of_story|>`) が必須 (ADR-021 で対応)

### 影響範囲
- `train_model.py`: `tokenize_function()` 後に `group_texts()` を適用
- データセット保存形式の変更 (Arrow)
