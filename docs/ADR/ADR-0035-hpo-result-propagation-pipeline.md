# ADR-0035: HPO結果確実反映パイプライン修正 (HPO-to-Full-Training Pipeline)

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Solo Developer

## Context

HPO探索で最適化されたハイパーパラメータが、フルスクラッチ学習（main.py）に確実に反映されていないことが判明。特に以下の3つの重大ギャップが存在した。

| # | ギャップ | 影響 |
|---|----------|------|
| 1 | **warmup_ratio 損失**: HPOで探索した5次元目のパラメータが `find_hparams.py` で固定値 0.03 に上書きされ破棄 | 5次元探索の意義が無効化 |
| 2 | **config.yaml 非同期**: `--target-model-size` 指定時、出力される `hparams_<target>.yaml` と `config.yaml` の `model.target_params` が不一致のまま | 実行時アーキテクチャ不一致リスク |
| 3 | **VRAM前提不整合**: プロキシVRAMで算出した batch 設定をターゲットモデル（異なるVRAM）に適用 | OOM または GPU 利用率低下 |

## Decision

### 1. warmup_ratio 尊重 (`find_hparams.py:325`)

```python
# 修正前
"warmup_ratio": 0.03,

# 修正後
"warmup_ratio": scaled_best.get("warmup_ratio", 0.03),
```

HPO探索結果（5次元目）を優先し、未探索時のみフォールバック。

### 2. `--sync-config` 機能追加 (`find_hparams.py`)

```bash
# 新引数
--target-vram-gb   # ターゲットVRAM指定（デフォルト: proxy VRAM）
--sync-config      # config.yaml も自動更新
```

`sync_config_yaml()` 関数で以下を自動更新：
- `model.target_params`
- `model.llama.*` (hidden_size, layers, heads, kv_heads, ffn, rope_theta)
- `defaults[0]` → `hparams_<target_size>`

### 3. ターゲットVRAM基準のバッチ計算 (`find_hparams.py:317`)

```python
# 修正前: proxy_vram 基準
per_device = min(target_batch_seqs, 1 if vram <= 4.5 else ...)

# 修正後: target_vram 基準
per_device = min(target_batch_seqs, 1 if target_vram <= 4.5 else ...)
```

### 4. 起動時整合性チェック (`config.py:_validate_config_consistency`)

`load_config()` 内で自動実行：
- `config.yaml` の `model.target_params` と `defaults: [hparams_XXX]` のモデルサイズ比較
- 10%以上乖離時 **WARNING** ログ出力
- 手動確認不要、毎回自動検証

## Consequences

### Pros
- **5次元探索の完全活用**: warmup_ratio も含め全パラメータが本番学習に反映
- **Proxy→Target 安全転移**: VRAM差異考慮、config.yaml 自動同期で人為ミス防止
- **実行時自動検証**: 設定不一致を早期発見、デバッグ時間短縮
- **CLI 完結**: `--sync-config` 一発で全整合性確保

### Cons
- `find_hparams.py` の引数増加（互換性維持のためデフォルト値で吸収）
- `sync_config_yaml()` は OmegaConf 依存（既存依存のため影響なし）

## Related ADRs
- ADR-0021: HPO Efficiency and Pruning
- ADR-0033: warmup_ratio → warmup_steps migration
- ADR-0034: Dynamic Search Space by Model Size

## 参照
- 概念的要件定義書: §4.1 責任分離マトリクス, §5.3 再開設計
- `scripts/find_hparams.py`
- `src/training/config.py`