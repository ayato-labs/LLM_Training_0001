# ADR-022: 推論テスト出力トレーサビリティ（構造化記録・クラウドバックアップ）

## ステータス
Accepted

## コンテキスト

### 現状の課題
論文の「Qualitative Results」セクションでは、モデルの出力例を示すことが必須である。現在のシステムでは：
- `evaluate_model.py` が推論テストを実行しMarkdownレポートを生成
- しかしMarkdownはローカル保存のみで、MLflowにもGoogle Driveにも記録されない
- 推論出力の再現が困難（モデルのバージョンと出力の対応関係が不明）
- 異なるチェックポイント間の出力比較が手動

### 要件
1. 推論テスト出力を構造化JSONとして記録
2. MLflow artifactとして記録（モデルと紐付け）
3. Google Driveに自動バックアップ
4. 再現スクリプトに推論コマンドを含める

---

## 意思決定

### 1. 構造化JSON出力
`evaluate_model.py` が以下の構造でJSONを生成する：

```json
[
  {
    "test_case_id": "TC-01",
    "test_case_name": "キャラクター知識と言葉遣いの検証",
    "target": "ジグの戦闘描写...",
    "prompt": "<|start_of_metadata|>...",
    "raw_output": "ジグは双刃剣...",
    "generated_output": "ジグは双刃剣...",
    "generation_params": {
      "max_new_tokens": 200,
      "temperature": 0.7,
      "top_p": 0.9
    },
    "model_path": "models/output",
    "timestamp": "2026-07-02T00:00:00"
  }
]
```

### 2. MLflow記録
- 各テストケースの出力を個別artifactとして記録
- Markdownレポートもartifactとして記録
- メトリクス: `inference_total_output_chars`, `inference_test_cases`

### 3. Google Driveバックアップ
`drive_uploader.py` が `logs/eval_report_*.md` と `logs/eval_results_*.json` を
`Novel_LLM_Inference_Reports/` フォルダに自動バックアップする。

### 4. モデルバックアップ
`drive_uploader.py` が `models/output/` をzip圧縮し、
`Novel_LLM_Models/` フォルダに時stamp付きでバックアップする。

---

## 結果と影響

### 1. 論文への貢献
- 「Qualitative Results」セクションの出力例が構造化して記録
- 異なるチェックポイント間の出力比較が容易
- 再現スクリプトに推論コマンドが含まれる

### 2. Google Drive容量（5TB）の活用
| アーティファクト | 推定サイズ | 保持方針 |
|-----------------|-----------|---------|
| チェックポイント zip | ~2.2GB × N個 | 全保持 |
| 最終モデル zip | ~600MB | 最新版のみ（上書き） |
| 推論レポート | ~100KB × N個 | 全保持 |
| MLflow/TensorBoard | ~50MB | 最新版のみ |
| DVCキャッシュ | <500MB | 最新版のみ |

合計: 1実験 series で約10-20GB消費（5TBに対して0.4%）

### 3. 関連ファイル
- `src/eval_inference/evaluate_model.py`: 推論テスト実行 + JSON記録
- `src/utils/drive_uploader.py`: モデル + 推論レポートのバックアップ
- `src/training/train_model.py`: モデルMLflow artifact記録
