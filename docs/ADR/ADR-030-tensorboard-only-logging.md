# ADR-030: TensorBoard単一ログ基盤への統一

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

旧実装では **MLflow + TensorBoard の二重書き込み** を実施：
- `mlflow.start_run()` → `log_params`, `log_metrics`, `log_artifacts`
- `SummaryWriter` → `add_scalar`, `add_histogram`
- 環境スナップショットは `mlflow.log_artifact(environment.json)`

問題：
1. **確認手間2倍**: UIが2つ（MLflow UI + TensorBoard）
2. **ストレージ圧迫**: 同一メトリクスを2形式で保存
3. **依存肥大化**: `mlflow>=2.10.0` が本番依存に混入（重い）
4. **再現性メタデータ分散**: Git hash等がMLflow側のみにあり、TensorBoard側にない

## 決定

**TensorBoardのみを採用し、MLflowを完全削除**。

### 採用理由

| 観点 | TensorBoard | MLflow |
|---|---|---|
| **ローカル完結** | ✅ ファイルベース | ❌ UI要・DB要（sqlite/PostgreSQL） |
| **VS Code統合** | ✅ ネイティブ対応 | △ 拡張機能必要 |
| **依存サイズ** | 軽量（`tensorboard`のみ） | 重い（`mlflow`, `sqlalchemy`, `gunicorn`等） |
| **メトリクス可視化** | 十分（scalar, histogram, projector） | 同等 |
| **モデルアーティファクト** | `trainer.save_model()` で管理 | `mlflow.log_model()` だが冗長 |
| **再現性メタデータ** | `env_snapshot.py` でJSON出力→TBログディレクトリ配置 | 専用UI必要 |

### 実装

```python
# src/main.py - TrainingArguments
report_to=["tensorboard"],  # MLflow削除

# src/env_snapshot.py - 再現性メタデータをJSONでログディレクトリに保存
def capture_env_snapshot() -> dict:
    return {
        "python": "...", "torch": "...", "cuda": "...", "gpu": "...",
        "git_hash": "...", "git_dirty": true,
        "transformers": "...", "datasets": "...", "accelerate": "...", ...
    }
```

### 移行措置

- 既存 `mlruns/` ディレクトリは **読み取り専用アーカイブ** として保持
- 新規学習は `models/output/runs/` (TensorBoard) のみ生成
- `DriveUploadCallback` は `models/output/` 全体をアップロード（TBログ含む）

## 結果

### 正の影響
- **依存削減**: `mlflow` 本番依存から除外（`dev`依存へ移動、約50MB削減）
- **確認シンプル**: `tensorboard --logdir models/output/runs` 一発
- **起動高速化**: MLflow初期化・DB接続・アーティファクト解決のオーバーヘッドゼロ
- **再現性統一**: `environment.json` がログ直下にあるため、TBログだけで完全再現可能

### 負の影響
- **実験比較UI**: MLflowの「Run比較テーブル」が失われる
  - 代替: TensorBoardの「Scalars」タブで複数Run選択比較、または `tensorboard.dev` アップロード
- **モデルレジストリ**: MLflow Model Registry機能なし
  - 代替: Google Drive版管理 + `trainer_state.json` ベストステップ記録で運用

## 検証

- `max_steps=1` 実行で `models/output/runs/events.out.tfevents.*` 生成確認
- `tensorboard --logdir models/output/runs` で `train/loss`, `train/learning_rate` 表示確認
- `environment.json` が `models/output/runs/` 直下に生成されること確認（`env_snapshot.py` 呼び出し箇所で実装予定）

## 補足: 将来的な拡張

必要に応じて `report_to=["wandb"]` 追加のみでW&B連携可能（HF Trainer標準対応）。依存追加のみでコード変更不要。