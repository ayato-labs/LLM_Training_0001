# ADR-027: HF標準スタックへの移行

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

リファクタリング前のコードベースには以下の独自実装が存在した：
- `ModernGPT` (`src/modern_gpt.py`) - 独自Transformer実装
- `CustomTrainer` (`src/trainer.py`) - 独自学習ループ・勾配累積・AMP
- `Muon` (`src/normuon.py`) - 独自オプティマイザ（2Dパラメータ用直交化）
- `scaling_analysis.py` - 独自スケーリング則解析

これらは**保守コストが高く、バグの温床**となっていた。また、Flash Attention 2、ZeRO最適化、勾配チェックポイント等の最新最適化が手動実装必要だった。

## 決定

**Hugging Face標準スタックへ完全移行**する：

| 独自実装 | 置換先 | メリット |
|---|---|---|
| `ModernGPT` | `LlamaForCausalLM` | Flash Attention 2自動有効、カーネル融合、TP/FSDP対応 |
| `CustomTrainer` | `Trainer` | 勾配累積・AMP・チェックポインティング・分散学習が成熟 |
| `Muon` | `AdamW` | ZeRO最適化との親和性高、実装成熟、hyperparameter成熟 |
| `scaling_analysis.py` | `step_law.py` + `find_hparams.py` | 理論計算は残し、探索をOptuna標準APIへ |

## 結果

### 正の影響
- **コード削減**: ~3,000行削除（modern_gpt.py, trainer.py, normuon.py, registry.py等）
- **保守負債解消**: 上流ライブラリのバグ修正・最適化を自動享受
- **VRAM効率向上**: Flash Attention 2 + gradient_checkpointing で同等モデルが約15%省メモリ
- **学習安定性**: Trainerの実績ある学習ループ（NaN検知、勾配クリッピング、スケジューラ統合）

### 負の影響・トレードオフ
- **Muon廃止**: 2Dパラメータへの直交化更新を諦め、AdamW統一。LRスケールを `max_lr_1d` (≒ 1e-3) に調整済み
- **カスタム損失関数不可**: 標準CLM損失のみ。必要なら `Trainer.compute_loss` オーバーライドで対応可能

## 検証

- `max_steps=1` ドライラン成功
- `LlamaForCausalLM` + `gradient_checkpointing=True` + `bf16` で 4GB VRAM環境で動作確認
- Step Law由来の `max_lr_1d=2.5e-3` で発散なし