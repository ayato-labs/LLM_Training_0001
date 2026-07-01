# ADR-020: 評価プロトコル標準化（複数シード・統計検定・自動レポート）

## ステータス
Accepted

## コンテキスト

### 現状の課題
論文レベルの実験評価には以下の要件が必須であるが、現在のシステムでは未対応である：
- **複数シード実行**: 単一シードでは結果の偶然性を排除できない
- **統計的検証**: 平均±標準偏差、信頼区間、有意差検定
- **横断比較**: 異なる設定間の定量的比較
- **自動レポート**: 論文の「Experimental Setup」セクションに必要な情報の自動生成

### 要件
1. 複数シード（≥3）で学習を実行し、結果を蓄積
2. 統計分析（平均、標準偏差、95% CI、t検定、Cohen's d）を自動計算
3. MLflowから実験結果を横断取得・比較
4. Markdown形式のレポートを自動生成

---

## 意思決定

### 1. 複数シード実行
**Hydra multirun** を使用して複数シードを並列実行する。

```bash
# 3シードで実行
python main.py --config-name=config -m seed=42,123,456

# 5シードで実行
python main.py --config-name=config -m seed=42,123,456,789,1024
```

- 各シードは独立したHydra jobとして実行
- 結果は `outputs/` に時刻別ディレクトリとして保存
- MLflowに各シードの結果が別々に記録

### 2. 統計分析
**`src/evaluation/statistics.py`** を新設し、以下の分析を提供：

| 分析手法 | 関数名 | 使用シーン |
|---------|--------|-----------|
| 要約統計 | `compute_summary()` | 平均/標準偏差/CI/範囲 |
| 対応のあるt検定 | `paired_t_test()` | 同一シード条件での比較 |
| Welch t検定 | `welch_t_test()` | 異なるシード条件での比較 |
| Cohen's d | `cohen_d()` | 効果サイズの測定 |
| グループ比較 | `compare_experiment_groups()` | 複数メトリクスの横断比較 |

### 3. 実験比較CLI
**`src/evaluation/compare_runs.py`** を新設：
- MLflowから実験結果を取得
- シード別にグループ化
- Markdown/JSONでレポート生成

```bash
# 直近20ランを比較
python -m src.evaluation.compare_runs --top 20

# シード別にグループ化
python -m src.evaluation.compare_runs --seed-group
```

### 4. 自動レポート生成
**`src/evaluation/report_generator.py`** を新設：
- 複数シードの結果から要約統計を自動計算
- 論文の「Experimental Setup」セクションに必要な情報を構造化
- 再現性チェックリストを自動生成

---

## 結果と影響

### 1. 論文への貢献
- 「Experimental Setup」セクションの情報が自動生成
- 「Results」セクションの表・統計が自動計算
- 「Reproducibility」の担保（全チェックリスト自動生成）

### 2. 計算コスト
- 3シード×3エポック ≈ 約200時間（RTX 3050 Laptop）
- 5シード×3エポック ≈ 約330時間
- Hydra multirun で並列実行可能（GPU数に依存）

### 3. 既存との整合性
- MLflow記録（ADR-018）と連携
- シード固定（ADR-016）と連携
- DVCデータ版管理（ADR-014）と連携

---

## 関連ファイル
- `src/evaluation/statistics.py`: 統計分析ユーティリティ
- `src/evaluation/report_generator.py`: レポート自動生成
- `src/evaluation/compare_runs.py`: MLflowラン比較CLI
- `configs/multi_seed.yaml`: Hydra multirun設定
