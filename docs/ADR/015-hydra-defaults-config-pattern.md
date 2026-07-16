# ADR-029: Hydra defaultsパターンによる設定構造化

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

旧設定構造：
- `config.yaml` (ベース)
- `scaling_50m.yaml`, `scaling_150m.yaml`, `scaling_3b.yaml`, `scaling_7b.yaml` (4種類)
- CLIで `--config scaling_150m.yaml` 指定必須
- 継承関係が不明確、`training` キーのネストが不統一

問題：
1. **どれを使うか不明**（4種類のscaling_*.yamlが並列存在）
2. **CLI引数必須**（デフォルトがない）
3. **上書き優先順位が暗黙的**（後勝ちだが順序依存）

## 決定

**Hydra `defaults` リスト + `_self_` + 単一hparamsアーティファクト** に統一。

### 設定ファイル構成（2ファイルのみ）

```
configs/
├── config.yaml          # ベース設定（変更稀）
└── hparams_150M.yaml    # 探索生成物（モデルサイズ固有の最適HP）
```

### `config.yaml` 定義

```yaml
# @package _global_
defaults:
  - hparams_150M   # 自動マージされる（_self_で自分も適用）

# 共通固定値（人間が変えることは稀）
seed: 42
precision: "bf16"
output_dir: "models/output"
data_path: "data/dataset.jsonl"
val_data_path: null
tokenizer_path: "data/tokenizer.json"

# デフォルト値（hparamsで上書きされる前提）
training:
  seq_len: 1024
  max_steps: -1
  num_epochs: 3
  save_steps: 1000
  eval_steps: 1000
  logging_steps: 10

# アーキテクチャ（スケール変更時のみ書き換え）
model:
  target_params: 150_000_000
  llama:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    num_key_value_heads: 3
    intermediate_size: 3072
    rope_theta: 10000.0
    vocab_size: 64000
```

### `hparams_150M.yaml` 定義（探索生成物）

```yaml
# @package _global_
training:
  max_lr_2d: 2.8e-4
  max_lr_1d: 2.5e-3
  batch_size_seqs: 16
  warmup_ratio: 0.035
  weight_decay: 0.1
  beta2: 0.95
  grad_clip: 1.0
per_device_batch_size: 1
grad_accum_steps: 16
```

### マージ挙動

1. `defaults: [hparams_150M]` → `hparams_150M.yaml` 読み込み
2. `config.yaml` 自身も適用（`_self_` 暗黙）
3. **同キーは後勝ち** → `hparams` の値が確実に優先される
4. ネストした `training:` キーも **深くマージ**（Hydraデフォルト）

## 実行時オーバーライド

```bash
# デフォルト（150M本番）
python -m src.main

# デバッグ短縮
python -m src.main +max_steps=100

# 別スケール（将来的に hparams_3B.yaml 作成時）
python -m src.main --config-name config ++defaults.0=hparams_3B
```

## 結果

### 正の影響
- **単一エントリーポイント**: `python -m src.main` で即動作
- **明確な優先順位**: `defaults` リスト順 = 適用順
- **拡張容易**: 新スケール追加 = `hparams_XXX.yaml` 追加のみ
- **型安全**: `config.py` で `OmegaConf.to_container(resolve=True, throw_on_missing=True)` による検証

### 負の影響
- Hydra学習コスト（初見は `defaults` / `_self_` / マージ挙動の理解必要）
- `_global_` パッケージ指定が必要（フラットマージのため）

## 検証

- `python -c "import hydra; hydra.initialize_config_dir('configs'); cfg=hydra.compose('config'); print(cfg)"` でマージ結果確認済み
- `max_lr_1d=2.5e-3` 等が `hparams_150M.yaml` 値で正しく上書きされること確認
- `throw_on_missing=True` で必須キー漏れを起動時に検知可能