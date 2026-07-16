# ADR-0033: warmup_ratio → warmup_steps 移行 (Transformers v5.2 非推奨対応)

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Solo Developer

## Context

HuggingFace Transformers で `warmup_ratio` パラメータが v5.2 で廃止予定という警告が発生。また、HPO探索空間に `warmup_ratio` が含まれておらず、最適なウォームアップ戦略が固定されていた。

## Decision

### 1. `warmup_ratio` → `warmup_steps` 完全移行

`train_engine.py` で `warmup_ratio` を `warmup_steps` に変換して使用：

```python
warmup_steps = hpo_config.get("warmup_steps", 0)
if warmup_steps == 0 and max_steps > 0:
    warmup_steps = int(max_steps * hpo_config.get("warmup_ratio", 0.03))
```

### 2. HPO探索空間に `warmup_ratio` 追加

`create_search_space()` を5次元に拡張：

```python
{
    "max_lr_2d": (center * 0.5, center * 2.0, "log"),
    "max_lr_1d": (center * 0.5, center * 2.0, "log"),
    "batch_size_seqs": [8, 16, 32],
    "weight_decay": (0.01, 0.3, ""),
    "warmup_ratio": (0.01, 0.1, ""),  # 1%〜10% の範囲で探索
}
```

### 3. 変換フロー

```
HPO: warmup_ratio (0.01〜0.1) を探索
  ↓
train_engine.py: warmup_steps = int(max_steps * warmup_ratio) に変換
  ↓
TrainingArguments: warmup_steps を使用
```

## Consequences

### Pros
- **Transformers v5.2 準備**: 非推奨警告が解消
- **HPO最適化**: ウォームップ率も探索対象となり、最適値を発見可能に
- **柔軟性**: モデルサイズ/データ量に応じた最適なウォームップ自動調整

### Cons
- 探索空間が4次元→5次元に増加（計算コスト若干増加）
- ただし warmup_ratio の探索範囲は狭く(0.01-0.1)、実質的な影響は最小

## 参照

- ADR-0021: HPO Efficiency and Pruning
- ADR-0026: Scratch Pre-training Optimizations
- 概念的要件定義書: §2.1 採用する標準スタック
