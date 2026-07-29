# ADR-0046: データセットトークン推定の精度改善 — JSONL text フィールド抽出 + 均一サンプリング

## ステータス
承認済み (Accepted)

## 日付
2026-07-29

## コンテキスト (Context)

Chinchilla Scaling Laws 計算機 (`src/scaling_laws/calculator.py:138-154`) が `detect_data_path_and_tokens()` (`src/scaling_laws/proxy_benchmark.py:19`) を呼び出し、データセットの総トークン数からモデルサイズの妥当性を検証している。

しかし、この関数の推定ロジックに以下の問題があり、**実際のトークン数を約1.75倍過大推定していた**：

### 問題1: 本番と異なるテキスト抽出

訓練パイプライン (`train_engine.py:469-470`) は JSONL から `"text"` フィールドのみを抽出して tokenize する。しかし、`detect_data_path_and_tokens()` は raw JSON 行（キー名 `{"text": "`, `"metadata": {...}}` などの構造を含む）をそのままサンプリングしていた。JSON 構造のバイト数が token/byte 比率を歪めていた。

### 問題2: 先頭バイアス

最初の200行（約200KB）のみをサンプリングしていた。JSONL ファイルの先頭にはメタデータのヘッダ行が偏在する可能性があり、ファイル全体の分布を代表しない。

### 問題3: バイトベース外挿による乖離

`total_tokens = file_size_bytes × (sample_tokens / sample_bytes)` の式で外挿していたが、JSON のキー構造を含むバイト数から比率を計算するため、実際の text 内容のみの token/char 比率と乖離していた。

### 実際の乖離

| 項目 | 旧推定値 | 実測値 (修正後) |
|------|---------|----------------|
| 総トークン数 | ~1,977M | ~1,128M |
| データ充足率 | 104.4% | ~60% |
| 警告表示 | なし | あり（不足） |

この過大推定により、Chinchilla 計算機はデータ不足を検出できず、実際より大きなモデルサイズ（86.5M）を推奨していた。

## 意思決定 (Decision)

`detect_data_path_and_tokens()` を、訓練パイプラインと同一のロジックに変更する：

### 1. テキスト抽出の本番一致

全行の JSON をパースし、`"text"` フィールドのみを抽出する。訓練パイプライン（`train_engine.py:469-470`）と同一の抽出方法。

### 2. 段階的推定

1. **第1パス**: 全行の `"text"` 文字数をカウント（高速、tokenize なし）
2. **第2パス**: 均一サンプリング（約2000章）で tokenize し、`token/char` 比率を算出
3. **推定**: `total_chars × (sample_tokens / sample_chars)`

### 3. サンプリングの均一化

従来の「先頭200行」から「全ファイルを等間隔で約2000サンプル」に変更。分布の偏りを排除。

### 変更ファイル

- `src/scaling_laws/proxy_benchmark.py` — `detect_data_path_and_tokens()` を全面改訂（`import json` 追加、抽出ロジックを本番一致に変更、サンプリングを均一化）

## 結果 (Consequences)

- **推定精度の改善**: 過大推定 (+75%) が解消され、実際のトークン数に一致
- **データ不足検出の正常化**: 86.5M モデルに対して Chinchilla 20x 基準でデータ不足を正しく警告
- **性能への影響**: 全行の JSON パース + 約2000章の tokenize により、従来より処理時間が増加（数分程度）するが、設定計算時に一度しか実行されないため許容範囲内
- **保守性**: 訓練パイプラインとのロジック統一により、推定値と実績値の乖離リスクが低減

## 参考ファイル

- `src/scaling_laws/proxy_benchmark.py` — `detect_data_path_and_tokens()` 修正 (lines 20-88)
- `src/training/train_engine.py` — 訓練時のテキスト抽出ロジック (lines 469-470)
- `src/training/model_utils.py` — 訓練時の tokenize 呼び出し (lines 203-212)
