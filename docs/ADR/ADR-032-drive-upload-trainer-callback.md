# ADR-032: Google Drive アップロードを Trainer Callback へ統合

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

旧実装では **独立デーモンプロセス** (`drive_uploader.py` の `monitor_and_upload()`) が 30秒ポーリングでチェックポイントを検知・圧縮・アップロードしていた。

問題：
1. **プロセス管理の複雑化**: 学習スクリプトとは別プロセスで起動・監視・再起動が必要
2. **タイミング不整合**: ポーリング間隔(30s)内にチェックポイント生成・削除が起きると取りこぼし
3. **ローカルクリーンアップ競合**: デーモンと学習プロセスでファイル削除競合
4. **Windows互換性**: `&` バックグラウンド実行・シグナルハンドリングが非標準

## 決定

**Hugging Face `TrainerCallback` として実装し、学習プロセス内で同期実行** する。

```python
# src/drive_uploader.py - DriveUploadCallback クラス
class DriveUploadCallback(TrainerCallback):
    def __init__(self, upload_interval_steps: int = 1000):
        self.upload_interval_steps = upload_interval_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.upload_interval_steps == 0:
            self._upload_checkpoint(state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        self._upload_checkpoint(state.global_step, force=True)
    
    def force_final_upload(self, global_step: int):
        self._upload_checkpoint(global_step, force=True)
```

`src/main.py` で登録：
```python
callbacks = [DriveUploadCallback(upload_interval_steps=config["drive_upload_interval"])]
trainer = Trainer(..., callbacks=callbacks)
```

## 代替案と却下理由

| 代替案 | 却下理由 |
|---|---|
| 従来通りデーモン継続 | プロセス管理負債が残る。Windowsで `nohup`/`systemd` 等不在 |
| DVC remote (GDrive) 使用 | DVCはデータバージョニング向け。チェックポイント頻出時のオーバーヘッド大 |
| `Trainer` の `push_to_hub` 機能流用 | HF Hub前提。GDrive API直叩きの方が制御可能 |

## 結果

### 正の影響
- **単一プロセス完結**: `python -m src.main` だけで学習＋アップロード完了
- **確実なタイミング**: ステップ終了フックで確実にアップロード（ポーリング不要）
- **例外安全**: Callback内 `try-except` でアップロード失敗しても学習継続
- **Windowsネイティブ動作**: 別プロセス・シグナル・デーモン化一切不要

### 負の影響
- **アップロード中ブロッキング**: 同期実行のためステップ間に遅延発生
  - 対処: `upload_interval_steps=1000` 以上推奨。必要なら `threading.Thread` で非同期化可能
- **大容量チェックポイント時のメモリ圧迫**: 圧縮時一時ファイル生成
  - 対処: `shutil.make_archive` ストリーミング書き込みでピーク抑制

## 運用ルール
- `drive_upload_interval` は `config.yaml` で設定（デフォルト 1000 step）
- 認証は `token.json` + `client_secret_*.json` （初回のみブラウザ認証）
- アップロード先: `Novel_LLM_Checkpoints/checkpoint-{step}.zip`
- 最終モデルは `on_train_end` で強制アップロード

## 検証
- `max_steps=1` で Callback 登録・初期化正常確認済み
- 実アップロードは認証ファイル配置後に手動検証予定