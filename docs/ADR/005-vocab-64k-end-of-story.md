# ADR-021: Vocab 64k + end_of_story Token 追加

- **Status:** Accepted
- **Date:** 2026-07-05
- **Deciders:** Novel LLM Team

## Context

現在の Tokenizer は vocab_size=32000 で、日本語小説ドメインでは OOV (Out-of-Vocabulary) 率 3-5% と大きい。
また、`<|start_of_story|>` は存在するが `<|end_of_story|>` がなく、物語の終了判定が曖昧。

## Decision

1. **Vocab を 64k に拡大**
2. **`<|end_of_story|>` を追加**

### 根拠

| vocab | OOV率 (小説) | 平均分割長 | embedding参数量 (100M) |
|-------|-------------|-----------|------------------------|
| 32k | 3-5% | 1.8-2.2 | 49M (49%) |
| **64k** | **1-2%** | **1.4-1.6** | **98M (98%)** |
| 128k | 0.5-1% | 1.2-1.3 | 197M (197%) |

- 64k で OOV 1-2% → 実質的に UNKNOWN トークンが出ない
- 128k は embedding が本体パラメータを食うため、100M では非推奨
- `<|end_of_story|>` は Packed Sequence で物語境界を明示するために必須

### 実装方針

1. Tokenizer 学習時に 64k vocab + 9特殊トークンで学習
2. DataPreprocessing 側で corpus.jsonl に `<|end_of_story|>` を付与
3. train_model.py で tokenizer.add_special_tokens() + model.resize_token_embeddings()

## Consequences

### メリット
- OOV 3-5% → 1-2% に減少
- 分割長 1.8-2.2 → 1.4-1.6 に短縮 → 同パラメータで実質的表現力向上
- `<|end_of_story|>` で物語終了を明示 → 生成制御・評価精度向上

### デメリット
- Tokenizer 再学習 + データ再トークナイズが必要 (1-2時間)
- embedding 参数量 49M → 98M に増加 (100M に対して 49% → 98%)
  - ただし 100M は PoC なので許容

### 影響範囲
- `train_tokenizer.py`: vocab_size=64000, special_tokens 9個
- `DataPreprocessing/src/cli.py`: `<|end_of_story|>` 付与
- `train_model.py`: tokenizer + model.resize_token_embeddings()
- `configs/config.yaml`: vocab_size: 64000
