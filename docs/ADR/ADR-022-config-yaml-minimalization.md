# ADR-022: config.yaml の最小化と responsibility 分離 / プロキシモデル動的スケーリング / スケーリング則ベース HPO拡張 / **Hydra defaults + オフラインHPOアーティファクト化**

- **Status:** Accepted
- **Date:** 2026-07-11
- **Updated:** 2026-07-12
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

**追加課題 (2026-07-12)**: Hydra未導入でCLIオーバーライドが不便。HPOが学習毎実行されリードタイム大。MLflow+TB二重ログ。

## Decision

config.yaml を30行の最小ベースに削減し、scaling_*.yaml は差分のみに。
**プロキシモデルの構成をパラメータ数から動的に生成するロジックを導入。**
**探索空間をスケーリング則 (Chinchilla, μP, MuTransfer) に基づき拡張。**
**【2026-07-12追加】Hydra defaults パターン + 単一 hparams_150M.yaml アーティファクト化。**

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

### 変更後 (Hydra defaults + 単一hparams)

**configs/config.yaml** (ベース・変更稀)
```yaml
# @package _global_
defaults:
  - hparams_150M   # 探索生成物で上書き

seed: 42
precision: "bf16"
output_dir: "models/output"
data_path: "data/dataset.jsonl"
val_data_path: null
tokenizer_path: "data/tokenizer.json"

training:
  seq_len: 1024
  max_steps: -1
  num_epochs: 3
  save_steps: 1000
  eval_steps: 1000
  logging_steps: 10
  drive_upload_interval: 1000

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

**configs/hparams_150M.yaml** (探索生成物・Git管理)
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

### 責任分離

| 設定項目 | 担当 | 理由 |
|----------|------|------|
| VRAM検出 | コード (`config.py`) | 環境依存で人間が設定不要 |
| precision (bf16) | コード (`config.py`) | GPU依存で人間が設定不要 |
| DeepSpeed設定 | コード (`model_utils.py`) | ZeRO stage は VRAM に依存 |
| gradient_checkpointing | コード (`main.py`) | VRAM に依存 |
| checkpoint保存間隔 | コード (`main.py`) | 過学習監視に最適化済み |
| ログ設定 | コード (`main.py`) | 固定値で十分 (TBのみ) |
| HPO探索 | **別ツール** (`scripts/find_hparams.py`) | 学習パイプラインから完全分離 |
| **モデルサイズ** | **YAML** (`config.yaml`) | 人間が実験ごとに変更 |
| **レイヤー構成** | **YAML** (`config.yaml`) | 人間が実験ごとに変更 |
| **最適HP** | **YAML** (`hparams_*.yaml`) | **探索ツールが生成、Git管理** |
| **seq_len** | **YAML** (`config.yaml`) | 人間がGPUメモリに合わせて変更 |
| **エポック数** | **YAML** (`config.yaml`) | 人間が過学習に合わせて変更 |

### プロキシモデル動的生成 (維持)

`src/model_utils.py` に `estimate_config_from_params(target_params)` を追加。

- **計算式**: `params = max(target_params * 0.05, 30_000_000)`
- **アーキテクチャ**: `L=12固定`, `H=12固定`, `H_dim` をパラメータ数から逆算
- **制約**: `n_embd % n_head == 0` を強制し、形状エラーを防止

`hpo_manager.py` の `objective()` がこの関数を呼び出し、本番モデル規模に追従したプロキシモデルで探索を実行。

### スケーリング則ベース HPO拡張 (維持)

`src/hpo_manager.py` に `compute_scaling_priors(target_params, n_tokens)` を追加。

- **Chinchilla 則**: 最適 LR ∝ model_size^(-0.35), 最適 batch ∝ model_size^0.5
- **μP 理論**: LR は 2D/1D 分離、grad_clip=1.0 がスケール不変で最適
- **MuTransfer**: 学習率の転移則でプロキシ→本番へ転移
- **探索空間拡張**: LR(2D/1D分離) + Weight Decay + Beta2 + Warmup + Grad Clip + Batch Size
- **Optuna 最適化**: TPE(multivariate=True) + MedianPruner で効率化

`hpo_manager.py` の `objective()` がこれらを用いて多次元探索を実行。

### 【2026-07-12追加】オフラインHPO・Hydra導入

```
scripts/find_hparams.py     ← 探索ツール（独立実行、数時間）
      │
      ├─ Step Law で Prior 計算
      ├─ Optuna で Proxy探索
      └─ 成果物出力 → configs/hparams_150M.yaml
                            │
                            ▼
src/main.py                 ← 学習パイプライン（即時開始）
      │
      └─ Hydraで config.yaml + hparams_150M.yaml を合成読み込み
         探索ロジック完全除去
```

- `optuna`, `mlflow` を `pyproject.toml` の `dev` 依存へ移動
- 学習起動リードタイム: 3-5分 → **<10秒**
- CLIオーバーライド: `python -m src.main +max_steps=100` 等 Hydra標準対応

## Consequences

### メリット
- YAMLの行数: 107行 → 30行 (72%削減) + hparams_150M.yaml ~20行
- 人間が触る必要のある設定だけが残る
- コードが自動処理する設定はYAMLから排除
- scaling_*.yaml 4種 → **hparams_150M.yaml 単一** (Git管理)
- **HPOプロキシモデルが本番モデルのスケールに自動追従し、探索結果が本番に転移可能になった**
- **スケーリング則に基づく事前分布で探索空間を絞り込み、試行回数を削減**
- **学習パイプラインから探索ロジック完全除去で、main.py ~200行、依存最小化、再現性向上**
- **HydraでCLIオーバーライド・設定合成・型検証が標準化**

### デメリット
- 設定を変更したい場合、コードを修正する必要がある (VRAM/DeepSpeed等)
- ただし、変更頻度の低い設定なので許容範囲
- 探索次元が増えた分、1試行あたりの計算コスト微増（MedianPrunerで補完）
- **初回/設定変更時**: 別途 `find_hparams.py` 実行必要（数時間）
- Hydra学習コスト (初見は `defaults` / `_self_` / マージ挙動の理解必要)

### 影響範囲
- `configs/config.yaml`: Hydra defaultsベースに書き換え
- `configs/hparams_150M.yaml`: **新規** (探索生成物、Git管理)
- `configs/scaling_*.yaml`: **削除** (4ファイル)
- `src/config.py`: Hydra DictConfig対応・フラット正規化・VRAM自動検出
- `src/main.py`: 設定読み込みをHydraベースに書き換え、探索ロジック完全除去
- `src/model_utils.py`: 維持 (プロキシモデル構成算出・DeepSpeed自動生成)
- `src/hpo_manager.py`: **ライブラリ化** (mainから非import、探索ロジックのみ)
- `scripts/find_hparams.py`: **新規作成** (オフラインHPO実行エントリーポイント)
- `pyproject.toml`: `optuna`, `mlflow` を `dev` 依存へ移動
- `run_hpo.bat`, `run_train.bat`: **新規作成** (Windowsランチャー)

---

## 関連 ADR

- **ADR-024**: パイロット自動化 (本番前検証)
- **ADR-023**: Windows ネイティブ安定化 (DLL 順序固定)
- **ADR-025**: スケーリング則ベースHPO探索空間の拡張 (Superseded by ADR-028)
- **ADR-026**: HPOにおけるVRAM自動検出と動的バッチサイズ調整
- **ADR-027**: HF標準スタックへの移行
- **ADR-028**: オフラインHPO分離
- **ADR-029**: Hydra defaults パターン設定構造化
- **ADR-030**: TensorBoard単一ログ基盤統一
- **ADR-031**: ローカル tokenizer.json 直接読み込み
- **ADR-032**: Drive アップロード Trainer Callback 統合
- **ADR-033**: Windows .bat ランチャー + uv 環境管理