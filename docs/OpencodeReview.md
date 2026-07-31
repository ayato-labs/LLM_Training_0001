# OpencodeReview レポート

- **対象**: LLM_Training (src/scaling_laws, src/hpo, src/training, src/common, src/context_extension, tests, scripts, tools)
- **環境**: torch 2.13.0+cu130 / transformers 5.13.1 / Python 3.12 / WSL2 Ubuntu
- **日付**: 2026-07-31
- **検証結果**: `ruff check` 22 エラー / `pytest` **23 failed, 63 passed, 1 skipped**（全モジュール import は成功）

---

## Critical

### 1. クロージャ遅延束縛バグ — 全層の Attention が最後の層の実装で実行される

`src/training/model_utils.py:110-127` (`apply_selective_attention_checkpointing`)

ループ内で `original_attn_forward` をクロージャにキャプチャしているため、**全ての層のラッパーが最後の層の `self_attn.forward`（=最後の層の重み）を呼び出す**。ruff B023 が検出、実際に実行して `[2,2,2]` で実証済み。

```python
for module in model.modules():
    original_attn_forward = module.self_attn.forward
    @wraps(original_attn_forward)
    def custom_attn_forward(*args, **kwargs):   # ← original_attn_forward は共有セル参照
        ...
```

→ 多層モデルで attention が全て最終層の重みで動く（破壊的）。`extension_engine.py:126` が呼ぶため context 拡張パスで発火。

**修正**: デフォルト引数で束縛する（`def custom_attn_forward(..., _fwd=original_attn_forward)`）か、`functools.partial` を使う。さらに `torch.utils.checkpoint._CheckpointFrame.check_recomputed_tensors_match = lambda ...: None` (L102-105) のグローバルモンキーパッチも副作用が広いので要再考。

---

## High

### 2. テストスイートが現在のコードと乖離（23件失敗）

- `tests/test_config_changes.py` — `src.config` (`load_config`/`resolve_config_path`) は存在しない（旧 `src/` フラット構成の残骸）。ADR パスも `docs/adr/`（小文字）を期待 → 実は `docs/ADR/`（大文字）。
- `tests/test_tokenizer_integration.py:468,475` — `src/train_tokenizer.py` が存在しない。`data/tokenizer_novel64k.model` (SentencePiece) も存在しない。
- `tests/generate_test.py` / `tests/test_150m.py` — コレクションエラー（`src.eval` / `main` が存在しない）。
- 63件は通過しているが、合格分の大半は旧スキーマ前提のテスト（例: `config.yaml` は現在 Hydra defaults 8行のみで `model.llama.hidden_size` 等は存在しない）。

### 3. HPO のトークン数推定が「行数」になっている

`src/hpo/main.py:90` — `n_tokens = sum(1 for _ in open(data_path))` は **JSONL の行数**。Step Law (`compute_hpo_for_target`) にトークン数として渡されるため、batch_size / LR の計算が桁で狂う。`scripts/calc_search_space.py:6` は `line_count * 1024` で補正しているのに対し本流は未補正。

**修正**: `src.scaling_laws.proxy_benchmark.detect_data_path_and_tokens()` を再利用するのが適切（`tools/analyze_dataset_tokens.py` も同様のロジックの重複あり）。

### 4. チェックポイント整合性検証が実質機能しない

`src/training/train_engine.py:340-345` — config ハッシュは `configs/config.yaml`（Hydra defaults の 8 行のみ）に対して計算されるが、実際の設定は `base_config.yaml` / `scaling_config.yaml` / `hpo_config.yaml` から合成される。**base_config や hpo_config を変更してもハッシュは変わらず、resume 時の「設定変更検知」が効かない**（トレーサビリティの核機能が盲点）。合成後コンテキストのシリアライズハッシュに変更すべき。

### 5. HPO プロキシ実行で `constant_cosine` スケジューラ指定が無視される

`src/hpo/hpo_manager.py:244-248` — `objective()` は `lr_scheduler_type: "constant_cosine"` を config **トップレベル**に置くが、`train_engine.py:541` は `config.get("hpo", config)` で **`hpo` サブ辞書（サンプリング済みパラメータのみ）** を読むため、デフォルト `"cosine"` にフォールバックし warmup/constant 指定も消失。設計意図（Step Law: constant+cosine）が反映されない。HPO トライアルの LR スケジュールが無意図の cosine になっている。

### 6. 評価が二重実行される

`train_engine.py:657-663` — `eval_strategy="steps"` + `eval_steps` で HF 標準評価が有効なのに、`PeriodicEvaluationCallback`（L692-700）が同じ間隔で `self.trainer.evaluate()` を再実行する（callbacks.py:315）。同じ step で 2 回評価 → 時間 2 倍。

---

## Medium

| # | 場所 | 内容 |
|---|------|------|
| 7 | `dual_optimizer_trainer.py:167` | `training_step` で `_prepare_inputs` を自前で呼んでから `super().training_step()` が再度呼ぶ（2重処理・device転送の重複）。カスタム計測は `_prepare_inputs` を借りず、`on_step_begin` などで代替可能 |
| 8 | `logger.py:153-159` | `LogContext` が一時オブジェクトの `contextualize().__enter__()` に入れて、**別インスタンス**で `__exit__` する → contextvar がリークし、`extra` が後続ログに残留し続ける |
| 9 | `train_engine.py:613-617` | epoch モード時の `estimated_total_steps = 10000` ハードコード。`len(train_ds)` から導出すべき |
| 10 | `proxy_benchmark.py:147` | `extract_throughput_from_recent_logs` が `tokens_per_it = 1024 * 32` をハードコード（実バッチと乖離すると tps が誤る）。ログに batch×seq を出すか設定から読むべき |
| 11 | `vram_estimator.py:458-466` | `_is_wsl()` は**未使用のデッドコード**で、`getattr(sys,"_called_from_test",...)` の複雑式が意味不明。`_calc_wsl_overhead_gb` も「WSL 用」のはずが全 Linux に 0.3GB 加算（誤り）。`estimate_training_vram` と `estimate_training_vram_with_calibration` はほぼ完全なコピペ重複（分岐一本化可能） |
| 12 | `critical_batch_law.py:60` / `common/vram_estimator.py` / `hpo_manager.py:147` | `calculate_tensor_core_mfu`・`_is_wsl`・`_wait_for_cuda_ready`（常に True を返すスタブ）等のデッドコードが複数 |
| 13 | `scripts/drive_uploader.py` | ADR-0025 で Google Drive 統合廃止済みのはずが 17KB のデーモンが残存。`cleanup_old_checkpoints` も `model_utils` と二重実装。削除 or 明示的な廃止宣言を |
| 14 | `src/rope/main.py` | CLI エントリポイント(`extend-context`)が TODO スタブのまま。本物は `context_extension` にある → 統合 or エントリポイント削除 |
| 15 | `muon.py:75` | `torch.jit.script` は torch 2.13 で **DeprecationWarning**（テスト実行時にも出力）。`torch.compile`/素の eager へ移行推奨。また `train_engine.py:330` の `capture_scalar_outputs = True` は torch 2.13 で設定自体が存在せず無効（属性を無意味に生やすだけ） |
| 16 | `set_seed.py:37` | `PYTHONHASHSEED` はインタプリタ起動後に設定しても無効。削除または設定が無効である旨をコメント化 |
| 17 | `config.py:218-279` | `_validate_config_consistency` は `hparams_*` 命名前提で、現在の Hydra defaults（`base_config`/`scaling_config`/`hpo_config`）では**常にスキップされ警告のみ**。旧構造の残骸 |
| 18 | `model_utils.py:407-424` | >100MB ファイルは「size+mtime」のメタデータハッシュに切り替え。mtime 同一で内容変更すると検知不能（resume 検証の意味が薄い）。あくまで目安である旨の明示を |

---

## Low / Style

- **ruff 22 エラー**（5 件は `--fix` 可能）: 主に N806/N812/N814（`muon.py` の NS 反復の変数名、`train_engine.py:48` の `F`、`extension_engine.py:93` の `_PTF`）、SIM108/102/105/115、F401（`benchmark_muon.py` の未使用 import）、I001、B023（上記 Critical 参照）。
- **改行コードのチャーン**: WSL 側 `git status` で 75 ファイル・+9247/-9247 の全行差分（Windows 側では clean 表示 → CRLF/LF の混在）。`core.autocrlf=false` かつ `.gitattributes` なし。**`.gitattributes` で `* text=auto eol=lf` を追加して一度正規化する**ことを推奨。
- 別マシン由来の `__pycache__`（`C:\Users\saiha\My_Service\...` パス）が混入していた（今回テスト前に削除）。`.gitignore` 徹底と定期削除を推奨。
- `pyproject.toml` の `requires-python = ">=3.12"` / torch 2.1+ と、実環境 torch 2.13 / transformers 5.13.1 の組み合わせは**検証済みか要確認**（特に HF Trainer API 変更、`LlamaForCausalLM`、`default_data_collator` 等の互換性。import は通るが train 実行は未検証）。

---

## 推奨アクション（優先順位）

1. **B023 クロージャバグの即時修正**（`model_utils.py`）+ 回帰テスト追加（2層モデルで層ごとの forward が別物であることを確認）
2. テストスイートの現状化（削除 or 新構成へ移行）— 23 失敗は CI を壊す
3. HPO のトークン数推定修正（`detect_data_path_and_tokens` 再利用）
4. config ハッシュの対象を合成設定全体へ
5. HPO プロキシのスケジューラ設定貫通（`hpo` サブ辞書に `lr_scheduler_type` 等を入れ、`train_engine` はトップレベルもフォールバックで読む）
6. 二重評価の解消、`LogContext` リーク修正
7. デッドコード整理（drive_uploader, rope/main, 未使用関数）+ ruff クリーンアップ
8. `.gitattributes` 追加と改行正規化
