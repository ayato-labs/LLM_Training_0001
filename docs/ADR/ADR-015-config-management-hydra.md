# ADR-015: 設定管理の統一（Hydra/OmegaConf導入）

## ステータス
Accepted

## コンテキスト

### 現状の課題
ADR-001で `config/config.yaml` への統一を決定したが、実際には以下の分散状態が継続していた：
- `training_config.py`（Python定数）
- `experiment_config.json`（探索フェーズ設定）
- `current_run_config.json`（本番学習設定）
- `generate_deepspeed_config()` による動的生成

これにより：
- 設定のバージョン管理が困難
- 実験間の設定比較が手動
- Hydraによる多重実行・スイープが利用できない

---

## 意思決定

**Hydra 1.3.3 + OmegaConf 2.3.1 を導入し、`configs/config.yaml` を単一の設定ソースとする。**

### 選定理由
| 基準 | Hydra | argparse+cattrs | 自作Config |
|------|-------|-----------------|-----------|
| CLI多重実行 | `python main.py -m seed=1,2,3` | なし | なし |
| 構造化設定 | OmegaConf DictConfig | dict | dataclass |
| 設定差分比較 | `hydra --cfg job` | 手動 | 手動 |
| 学習コスト | 中（既存コード改修） | 低 | 高 |
| ライセンス | MIT | - | - |

### 設定構造（`configs/config.yaml`）
```yaml
hardware:    # VRAM制限、精度
data:        # パス、前処理パラメータ
model:       # アーキテクチャ、パラメータ数
tokenizer:   # vocab_size、特殊トークン
training:    # seq_len、エポック、HPO、オプティマイザ
seed:        # 乱数シード
mlflow:      # トラッキングURI、実験名
checkpoint:  # 保存戦略
logging:     # レポート先
```

### 後方互換性
- `main.py --legacy` で従来の `training_config.py` ベースの動作を維持
- `main.py --config-name=config` でHydraモードを有効化
- JSON config互換: `normalize_config()` が DictConfig/JSON 両方に対応

---

## 結果と影響

### 1. トレーサビリティ効果
- 1実験 = 1つのYAMLスナップショットがMLflowに記録
- CLIオーバーライドで設定変更が履歴に残る
- `python main.py --cfg job` で実行時設定を確認可能

### 2. 多重実行（将来Phase 2）
```bash
# 複数シードで並列実行
python main.py -m seed=42,123,456

# LRスイープ
python main.py -m training.hpo.max_lr_2d=0.001,0.005,0.01
```

### 3. 制約
- 既存の `training_config.py` はPhase 1では廃止せず（後方互換維持）
- 将来Phase 2で完全移行を検討
