# ADR-0043: Step Law 準拠アーキテクチャへの修正（Muon実装とHPOスケーリング整合性確保）

- **Status:** Accepted
- **Date:** 2026-07-20
- **Deciders:** Ayato-labs (ayato-labs)

## Context

従来のアーキテクチャにおいて、Step Law (arXiv:2503.04715) に基づくスケーリング則を体現できていない以下の問題が判明した：

### 1. Muon Optimizer 未実装（最重要）
Step Law の最適LR式 `η* = 1.79 * N^(-0.713) * D^(0.307)` は **Muon optimizer** を前提として導出されている。
しかし、本プロジェクトでは `adamw_bnb_8bit` 単一オプティマイザで全パラメータを最適化しており、2D/1Dパラメータの学習率分離（`max_lr_2d` vs `max_lr_1d`）が機能していなかった。

### 2. HPOスケーリングの実装場所が不明確
`scripts/find_hparams.py` でスケーリングを実行する設計（ADR-014）だが、`train_engine.py` 側でも暗黙的にスケーリングを期待するコードが混在し、責任境界が曖昧だった。

### 3. 設定の不整合
- `configs/hparams_150M.yaml` に `max_lr_2d` / `max_lr_1d` が定義済みにもかかわらず、実際には単一 `learning_rate` として AdamW に渡されていた
- `TrainingArguments.learning_rate` には `max_lr_2d` が設定されていたが、1Dパラメータにも同一LRが適用されていた

---

## Decision

以下の3点を実装し、Step Law準拠アーキテクチャへ修正する：

### 1. Muon + AdamW 1D 分離最適化の実装

```
SplitOptimizer (新規)
├── Muon (2D params: weight matrices)     ← lr = max_lr_2d
└── AdamW 8bit (1D params: embed/bias/LN) ← lr = max_lr_1d
```

- `src/training/optimizers/muon.py`: Newton-Schulz直交化（5反復）実装
- `src/training/optimizers/split_optimizer.py`: パラメータ分離・複合最適化
- `src/training/trainer/dual_optimizer_trainer.py`: HF Trainer拡張

### 2. HPOスケーリング責任の明確化

| フェーズ | 責任 | 実装場所 |
|---------|------|---------|
| **探索 (Offline)** | Step Law Prior計算 → Proxy探索 → Targetスケーリング → YAML出力 | `scripts/find_hparams.py` |
| **学習 (Online)** | スケーリング済みYAMLを読み込み、即実行（計算なし） | `src/training/train_engine.py` |

**不変条件**: `train_engine.py` にはスケーリングロジックを一切置かない。

### 3. 設定整合性の強制

- `config.py` にて `model.target_params` と `hparams_*.yaml` のモデルサイズ整合性を起動時に検証（既存 `_validate_config_consistency` 強化）
- 学習率キー命名統一: `max_lr_2d` (Muon), `max_lr_1d` (AdamW)

---

## Consequences

### Pros
### Positive 

- **Step Law理論準拠**: 2D/1D学習率分離により、スケーリング指数 `-0.713` が正しく機能
- **スケーラビリティ確保**: 150M→3B→7B への拡張時に、HPO再実行のみで最適LR自動導出可能
- **VRAM効率**: AdamW 8bit は1Dパラメータのみ対象 → VRAM節約効果維持
- **責任分離**: 探索/学習のフェーズ分離（ADR-014）がコードレベルで強制される

### Cons

- **実装複雑度増**: カスタムTrainer・分離Optimizerの保守コスト発生（~200行追加）
- **チェックポイント互換性**: 既存 `adamw_bnb_8bit` チェックポイントからのレジューム不可（要スクラッチ再開）
- **bitsandbytes依存**: Windowsネイティブでは8bit AdamW不可（WSL2必須）

### Risks & Mitigations

| リスク | 影響度 | 対策 |
|-------|-------|------|
| Muon実装バグ | 高 | 100 step 検証実行・loss曲線確認 |
| LRスケーリング係数誤り | 高 | Step Law原論文式との単体テスト追加 |
| 既存チェックポイント破棄 | 中 | 破棄前提で運用・ドキュメント化 |

---

## Implementation Details

### ファイル構成追加

```
src/training/
├── optimizers/
│   ├── __init__.py
│   ├── muon.py              # Muon optimizer (Newton-Schulz 5 iter)
│   └── split_optimizer.py   # Muon(2D) + AdamW8bit(1D) 複合
└── trainer/
    ├── __init__.py
    └── dual_optimizer_trainer.py  # Trainer拡張
```

### 設定例（hparams_150M.yaml → 3Bスケーリング後）

```yaml
# 150M Proxy HPO結果
training:
  max_lr_2d: 0.0005
  max_lr_1d: 0.0003
  batch_size_seqs: 32
  weight_decay: 0.168
  ...

# 3B Target へスケーリング (ratio = (3e9/150e6)^-0.713 ≈ 0.074)
training:
  max_lr_2d: 0.000037  # 0.0005 * 0.074
  max_lr_1d: 0.000022  # 0.0003 * 0.074
  batch_size_seqs: 32  # 不変 (Step Law: B* ∝ D^0.571, N非依存)
  weight_decay: 0.168  # 不変 (経験的)
  ...
```

### 移行手順

1. 既存チェックポイント破棄（互換性なし）
2. HPO再実行: `run_hpo.bat 150M ... --target-size 3B --sync-config`
3. 学習実行: `run_train.bat`（`hparams_3B.yaml` 自動読み込み）

---

## References

- ADR-0014: オフラインHPO（探索フェーズ分離）
- ADR-0028: 事前学習最適化（Liger Kernel, torch.compile, SDPA）
- ADR-0030: HPO Proxy Scaling Transfer
- ADR-0040: ハードウェア適応型最適化ポリシー
- Step Law Paper: arXiv:2503.04715
- Muon Optimizer: https://github.com/KellerJordan/Muon