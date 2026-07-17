# ADR-0036: env_snapshot.py 廃止 (Environment Snapshot Removal)

- **Status:** Accepted
- **Date:** 2026-07-17
- **Deciders:** Solo Developer

## Context

`src/common/env_snapshot.py` は学習起動時に Git ハッシュ、Python/Torch/CUDA バージョン、GPU 名等を収集し `environment.json` として出力する機能を提供していた。しかし以下の問題があった。

1. **未実装**: 概念的要件定義書 §6.1 では `models/output/runs/environment.json` へ自動保存と記載されていたが、実際は `logger.debug()` でコンソール出力するのみでファイル保存されていなかった
2. **冗長性**: Git ハッシュ等は `git log --oneline -1` や CI/CD パイプラインで取得可能。学習ログ（TensorBoard）と重複
3. **保守コスト**: 独立ファイル・関数として維持する価値が薄い（約50行）
4. **実用性不足**: 再現性確保の核心は「設定とデータのハッシュ検証（§5.3）」であり、環境メタデータは補助的

## Decision

`env_snapshot.py` を削除し、関連コードを整理する。

### 削除対象
- `src/common/env_snapshot.py` (ファイル全体)
- `src/training/train_engine.py` の import と呼び出し (3行)

### 残すもの
- `HashSaveCallback` による `config_hash` / `data_hash` 検証（再開時の整合性保証の核心）
- `Trainer` による `config.yaml` 自動保存（Hydra合成済み設定の記録）

## Consequences

### Pros
- コードベース簡素化（-50行、-1ファイル）
- 実装と要件定義書の齟齬解消
- 保守対象減少

### Cons
- 学習ログに環境バージョンが自動記録されなくなる（必要なら `pip freeze` / `git describe` をCIで別途記録）

## Related
- 概念的要件定義書 §6.1 (削除), §8 (許可独自実装から削除)
- ADR-0021: HPO Efficiency and Pruning (no direct relation)

## 参照
- `src/training/train_engine.py`: `HashSaveCallback` による厳格ハッシュ検証が継続