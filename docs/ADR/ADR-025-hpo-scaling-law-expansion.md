# ADR-025: スケーリング則に基づく HPO 探索空間の拡張と効率化

- **Status:** Accepted
- **Date:** 2026-07-11
- **Deciders:** Novel LLM Team

## Context

従来の HPO (Optuna) は「学習率のみ」を探索しており、以下の問題があった：

1. **探索次元の不足**: LR 以外のハイパーパラメータ（Weight Decay, Beta2, Warmup, Grad Clip, Batch Size）が固定されており、相互作用を考慮できていなかった
2. **探索範囲の非効率**: 経験則的な広い範囲（例: 1e-5 〜 1e-2）を log-uniform で探索しており、明らかに非効率な領域まで探索していた
3. **スケーリング則の未活用**: Chinchilla 則・μP (μTransfer) 等の確立されたスケーリング則を活用しておらず、プロキシモデル (30M) で得られた知見を本番 (150M) に転移する理論的根拠が弱かった
3. **探索アルゴリズムの未最適化**: デフォルトの TPE sampler (単変量) / Pruner なしで、相関学習・早期打ち切りが機能していなかった

## Decision

### 1. スケーリング則に基づく事前分布の導入 (`compute_scaling_priors`)

Chinchilla 則・μP (Maximal Update Parametrization) に基づき、目標パラメータ数 `N` とトークン数 `D` から理論的最適値を算出し、その周辺を探索範囲とする。

| ハイパラ | スケーリング則 | 実装式 | 探索範囲 |
|----------|--------------|--------|----------|
| LR (2D/1D) | Chinchilla + μP | `lr ∝ N^{-0.35}`, 2D/1D分離 | 中心値 × [0.3, 3.0] (log-uniform) |
| Batch Size | Chinchilla | `bs ∝ N^0.5` | [8, 16, 32, 64] (VRAM制約で上限) |
| Weight Decay | 経験則/μP | `wd ∝ N^{-0.1}` | [0.01, 0.3] (log-uniform) |
| Beta2 | μP | 大きいモデルで 0.95 | [0.95, 0.98, 0.99, 0.999] |
| Warmup Ratio | 経験則 | `≈ 0.03` 固定 | [0.01, 0.1] (log-uniform) |
| Grad Clip | μP | `clip=1.0` (スケール不変) | [0.5, 2.0] |

### 2. LR の 2D/1D 分離 (μP / Maximal Update Parametrization)

- **2D params** (Attention QKV, FFN weights): `lr_2d ∝ 1/width`
- **1D params** (Bias, LayerNorm, Embeddings): `lr_1d ∝ 1`
- プロキシ (30M) で最適 LR 比率を見つけ、本番 (150M) へ μTransfer で転移可能

### 3. 多変量 TPE Sampler + MedianPruner

- `TPESampler(multivariate=True)` でパラメータ間相関を学習
- `MedianPruner(n_startup_trials=3, n_warmup_steps=10)` で明らかに悪い試行を早期打ち切り

### 4. 探索パラメータの拡張

従来: `lr` のみ  
拡張後: `lr_2d`, `lr_1d`, `weight_decay`, `beta2`, `warmup_ratio`, `grad_clip`, `batch_size`

---

## Consequences

### メリット
- **探索効率の劇的向上**: 理論的最適値周辺 (±0.3〜3倍) に探索空間を絞り込み、無駄な試行を排除
- **本番への転移可能性**: μP 理論に基づく LR 分離により、プロキシ (30M) で見つけた最適 LR 比率が本番 (150M) に転移可能
- **早期打ち切り**: `MedianPruner` で明らかに悪い試行を 10 step で打ち切り、計算資源を節約
- **多変量相関学習**: `multivariate=True` で LR と WD、LR と BS 等の相関を学習し、効率的な探索軸を自動発見

### デメリット
- 実装の複雑化（`hpo_manager.py` が大幅拡張）
- 探索次元増加により、収束までの試行数が増える可能性（Pruner で緩和）

---

## 影響範囲

- `src/hpo_manager.py`: 全面改修 (`compute_scaling_priors`, `objective`, `main`)
- `src/model_utils.py`: 既存の動的スケーリングと整合
- `src/main.py`: 既存のパイロット自動化と整合

---

## 関連 ADR

- **ADR-022**: プロキシモデル動的スケーリング (5% / 下限30M)
- **ADR-024**: パイロット自動化 (本番前検証)
- **ADR-023**: Windows ネイティブ安定化 (DLL 順序固定)