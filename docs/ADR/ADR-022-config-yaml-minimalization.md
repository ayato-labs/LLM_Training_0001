# ADR-022: config.yaml の最小化と responsibility 分離 / プロキシモデル動的スケーリング / スケーリング則ベース HPO拡張

- **Status:** Accepted
- **Date:** 2026-07-11
- **Updated:** 2026-07-11
- **Deciders:** Novel LLM Team

## Context

以前の `config.yaml` は107行で、以下のような問題があった:

1. **設定とコードの責任混在**
   - VRAM検出、precision選択、DeepSpeed設定 → コードがやるべきこと
   - モデルサイズ、seq_len、エポック数 → 人間が変更すること
   - この2つが混在していた
2. **YAMLの肥大化**
   - `hardware`, `deepspeed`, `checkpointing` セクションは全てコードが自動処理
   - 人間が触らない設定が80%を占めていた
3. **scaling_*.yaml の冗長性**
   - ベース設定を丸ごとコピー → 差分だけ記述すべき

**追加課題 (2026-07-11)**: HPO (Optuna) で使用するプロキシモデルのサイズがハードコードされており、本番モデルのスケールに追従しなかった。また、探索空間が LR のみで、スケーリング則を活用した効率的な探索ができていなかった。

## Decision

config.yaml を30行の最小ベースに削減し、scaling_*.yaml は差分のみに。
**プロキシモデルの構成をパラメータ数から動的に生成するロジックを導入。**
**探索空間をスケーリング則 (Chinchilla, μP, MuTransfer) に基づき拡張。**

### 変更前 (107行)
```yaml
# config.yaml (旧)
seed: 42
model:
  arch: "llama"
  target_params: 150_000_000
  llama:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    num_key_value_heads: 3
    intermediate_size: 3072
    rope_theta: 10000.0
  data:
    dataset_path: "data/dataset.jsonl"
    tokenizer_path: "data/tokenizer.json"
  training:
    seq_len: 1024
    max_steps: -1
    num_epochs: 3
  hardware:
    vram_gb: 4
    precision: "bf16"
    auto_detect_vram: true
  deepspeed:
    enabled: true
    zero_stage: 1
    offload_optimizer: false
    offload_param: false
  checkpointing:
    gradient_checkpointing: true
    save_steps: 500
    save_total_limit: 3
  logging:
    logging_steps: 10
    tensorboard: true
    log_dir: "logs"
```

### 変更後 (30行)
```yaml
# config.yaml (新)
seed: 42
model:
  arch: "llama"
  target_params: 150_000_000
  llama:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    num_key_value_heads: 3
    intermediate_size: 3072
    rope_theta: 10000.0
data:
  dataset_path: "data/dataset.jsonl"
  tokenizer_path: "data/tokenizer.json"
training:
  seq_len: 1024
  max_steps: -1
  num_epochs: 3
```

### 責任分離

| 設定項目 | 担当 | 理由 |
|----------|------|------|
| VRAM検出 | コード (`config.py`) | 環境依存で人間が設定不要 |
| precision (bf16) | コード (`config.py`) | GPU依存で人間が設定不要 |
| DeepSpeed設定 | コード (`train_model.py`) | ZeRO stage は VRAM に依存 |
| gradient_checkpointing | コード (`train_model.py`) | VRAM に依存 |
| checkpoint保存間隔 | コード (`train_model.py`) | 過学習監視に最適化済み |
| ログ設定 | コード (`logger.py`) | 固定値で十分 |
| **モデルサイズ** | **YAML** | 人間が実験ごとに変更 |
| **レイヤー構成** | **YAML** | 人間が実験ごとに変更 |
| **seq_len** | **YAML** | 人間がGPUメモリに合わせて変更 |
| **エポック数** | **YAML** | 人間が過学習に合わせて変更 |

### プロキシモデル動的生成 (追加)

`src/model_utils.py` に `estimate_config_from_params(target_params)` を追加。

- **計算式**: `params = max(target_params * 0.05, 30_000_000)`
- **アーキテクチャ**: `L=12固定`, `H=12固定`, `H_dim` をパラメータ数から逆算
- **制約**: `n_embd % n_head == 0` を強制し、形状エラーを防止

`hpo_manager.py` の `objective()` がこの関数を呼び出し、本番モデル規模に追従したプロキシモデルで探索を実行。

### スケーリング則ベース HPO拡張 (追加)

`src/hpo_manager.py` に `compute_scaling_priors(target_params, n_tokens)` を追加。

- **Chinchilla 則**: 最適 LR ∝ model_size^(-0.35), 最適 batch ∝ model_size^0.5
- **μP 理論**: LR は 2D/1D 分離、grad_clip=1.0 がスケール不変で最適
- **MuTransfer**: 学習率の転移則でプロキシ→本番へ転移
- **探索空間拡張**: LR(2D/1D分離) + Weight Decay + Beta2 + Warmup + Grad Clip + Batch Size
- **Optuna 最適化**: TPE(multivariate=True) + MedianPruner で効率化

`hpo_manager.py` の `objective()` がこれらを用いて多次元探索を実行。

## Consequences

### メリット
- YAMLの行数: 107行 → 30行 (72%削減)
- 人間が触る必要のある設定だけが残る
- コードが自動処理する設定はYAMLから排除
- scaling_*.yaml は差分だけ記述 → 冗長性の排除
- **HPOプロキシモデルが本番モデルのスケールに自動追従し、探索結果が本番に転移可能になった**
- **スケーリング則に基づく事前分布で探索空間を絞り込み、試行回数を削減**

### デメリット
- 設定を変更したい場合、コードを修正する必要がある
- ただし、変更頻度の低い設定なので許容範囲
- 探索次元が増えた分、1試行あたりの計算コスト微増（MedianPrunerで補完）

### 影響範囲
- `configs/config.yaml`: 最小ベースに書き換え
- `configs/scaling_*.yaml`: 差分のみに削減
- `src/config.py`: 新規作成 (YAML読み込み + VRAM自動検出)
- `src/main.py`: 設定読み込みをYAMLベースに書き換え
- `src/model_utils.py`: 新規作成 (プロキシモデル構成算出)
- `src/hpo_manager.py`: 動的プロキシ生成 + スケーリング則ベース探索に対応
