# ADR-028: オフラインHPO（探索フェーズの分離）

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

旧実装では `main.py` 起動時に毎回以下を実行していた：
1. Optuna Study作成（`n_trials=10-20`）
2. Proxy学習（50 steps × 0.5% data）を試行毎に実行
3. Step Law適用
4. 本番学習開始

**問題**: 学習開始まで **3-5分のリードタイム** が発生。実験回転率が極端に低下。

## 決定

**探索フェーズ（HPO）を完全に分離し、成果物をYAMLアーティファクトとして保存**する。

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
      └─ hparams.yaml 読み込みのみ（探索ロジックゼロ）
```

**成果物フォーマット** (`configs/hparams_150M.yaml`)：
```yaml
training:
  max_lr_1d: 2.5e-3       # AdamW用LR（Step Law由来 + Optuna最適化）
  max_lr_2d: 2.8e-4       # 互換性のため保持
  batch_size_seqs: 16
  warmup_ratio: 0.035
  weight_decay: 0.1
  beta2: 0.95
  grad_clip: 1.0
per_device_batch_size: 1  # 派生値（VRAM依存）
grad_accum_steps: 16      # 派生値
```

## 代替案と却下理由

| 代替案 | 却下理由 |
|---|---|
| 学習時もOptuna実行（キャッシュ併用） | キャッシュ無効化・環境差異で再現性リスク。リードタイム改善せず |
| MLflow/W&BでHPO管理 | 依存追加・サーバー不要の原則に反する。ローカルYAMLで十分 |
| 事前計算をコードにハードコード | モデルサイズ・データ量・VRAM変更時に手動修正必要。DRY違反 |

## 結果

### 正の影響
- **学習開始リードタイム**: 3-5分 → **<10秒**（設定読み込みのみ）
- **再現性**: `hparams.yaml` + `git hash` で完全固定
- **CI/CD親和性**: `find_hparams.py` を夜間バッチで実行、成果物をアーティファクト保存可能
- **単一責任**: `main.py` は「学習のみ」に集中（~200行）

### 負の影響
- **初回/設定変更時**: 別途 `find_hparams.py` 実行必要（数時間）
- **運用フロー追加**: 開発者は「探索→学習」の2ステップを理解必要

## 運用ルール

1. **モデルサイズ・データ量・VRAM変更時**のみ `find_hparams.py` 再実行
2. `hparams_*.yaml` は **Git管理**（再現性のため）
3. 学習スクリプトから `optuna`, `mlflow` import **完全排除**（本番依存から除外）

## 検証

- `python -m scripts.find_hparams.py --model-size 150M --n-trials 3` 正常完了
- 生成された `hparams_150M.yaml` を `main.py` が正常読み込み
- `optuna` `mlflow` 未インストール環境で `main.py` 単体動作確認済み