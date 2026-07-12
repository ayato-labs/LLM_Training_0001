# ADR-026: HPOにおけるVRAM自動検出と動的バッチサイズ調整

- **Status:** Accepted
- **Date:** 2026-07-12
- **Deciders:** Novel LLM Team

## Context

ADR-025 で導入されたスケーリング則ベースのHPO探索空間において、バッチサイズの探索候補がVRAM 4GB固定で `[1, 2, 4]` にハードコードされていた（`hpo_manager.py:174-178`、`compute_scaling_priors`内）。

これにより以下の問題があった：

1. **ハードウェア非依存性の欠如**: 8GB/12GB/16GB/24GB以上のGPU環境でも、4GB想定の小さなバッチサイズしか試せず、効率的な探索ができない
2. **手動設定の手間**: 環境ごとにコード修正が必要
3. **スケーリング則との不整合**: 理論的最適バッチサイズ（`bs ∝ N^0.5`）がVRAM制約でクリップされる仕組みが動的でない

既存の `src/config.detect_vram()` が `torch.cuda.get_device_properties()` で自動検出（フォールバック 4.0GB）を実装済みだったため、これを活用可能。

## Decision

### 1. VRAM自動検出の統合
`src/config.detect_vram()` を `hpo_manager.py` にインポートし、HPO実行時に自動的にGPU VRAMを検出する。

### 2. 動的バッチサイズ候補生成関数 `_get_batch_size_candidates(vram_gb)` の導入

| VRAM容量 | バッチサイズ候補 | 適用シーン |
|----------|-----------------|------------|
| ≥ 24GB | `[4, 8, 16, 32]` | A100/H100等 |
| ≥ 16GB | `[2, 4, 8, 16]` | RTX 4080/3090等 |
| ≥ 12GB | `[2, 4, 8]` | RTX 3060 12GB等 |
| ≥ 8GB | `[1, 2, 4, 8]` | RTX 3070/4070等 |
| ≤ 4GB | `[1, 2, 4]` | 既存フォールバック（従来通り） |

### 3. `compute_scaling_priors` の拡張
- 第3引数 `vram_gb: float = 4.0` を追加
- 理論的最適バッチサイズ (`optimal_batch`) と VRAM制約上限 (`max(batch_size_candidates)`) の最小値を事前分布 `batch_size_prior` として採用

### 4. `objective` 関数での統合
```python
vram_gb = config.get("vram_limit_gb", detect_vram())
priors = compute_scaling_priors(target_params, n_tokens, vram_gb)
batch_size = trial.suggest_categorical("batch_size", priors["batch_size_candidates"])
```

### 5. 重複コードの整理
`hpo_manager.py` 内に混在していた `objective` 関数 4重定義、`_cleanup_vram` 3重定義を単一化。

## Consequences

### メリット
- **完全自動化**: GPU交換・環境移行時のコード修正不要
- **探索効率最大化**: 利用可能なVRAMに応じて最大バッチサイズまで探索可能
- **理論的整合性**: スケーリング則の最適値とVRAM制約のバランスを動的に計算
- **後方互換性**: VRAM検出失敗時は 4GB として安全側にフォールバック

### デメリット
- 初回実行時の `torch.cuda.get_device_properties()` 呼び出しオーバーヘッド（微小）
- 将来的にCPU-only環境対応が必要な場合は検出ロジックの拡張が必要

---

## 影響範囲

- `src/hpo_manager.py`: 全面リファクタリング（インポート追加、関数追加、2関数修正、重複除去）
- `src/config.py`: 既存 `detect_vram()` を再利用（変更なし）

---

## 関連 ADR

- **ADR-025**: スケーリング則ベースHPO探索空間の拡張（本ADRの前提）
- **ADR-022**: プロキシモデル動的スケーリング（5% / 下限5M）
- **ADR-024**: パイロット自動化（本番前検証）