# ADR-033: Windows ネイティブ起動スクリプト（.bat）と uv 環境管理の採用

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

リファクタリング前は `python -m src.main` 直叩き・仮想環境手動アクティベート・依存 `pip install -e .` 手動実行だった。

問題：
1. **Windows PowerShell で `&&` `||` `source` が使えない** → 手順書が煩雑
2. **仮想環境パスがプロジェクト外** (`DataPreprocessing/.venv`) で `uv` デフォルト挙動と衝突
3. **依存再インストール頻発**: ディスクフル・キャッシュ破損時のリカバリ手順が属人化
4. **Hydra オーバーライド記法** (`+max_steps=1`) がシェルエスケープで面倒

## 決定

**`.bat` ランチャー + `uv` 固定環境** で標準化。

### ファイル構成

```
LLM_Training/
├── train.bat          # 本番学習
├── train_debug.bat    # デバッグ短縮
├── train_resume.bat   # 再開
├── hpo.bat            # オフラインHPO実行
└── setup_env.bat      # 環境構築・修復
```

### `train.bat` 例

```bat
@echo off
REM ============================================================
REM Novel LLM Training Launcher (Production)
REM Usage: train.bat [Hydra overrides...]
REM   train.bat
REM   train.bat max_steps=1000
REM   train.bat +experiment=debug
REM ============================================================

cd /d "%~dp0"

REM 環境チェック・自動修復
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual env not found. Running setup_env.bat...
    call setup_env.bat
    if errorlevel 1 (
        echo [ERROR] Environment setup failed.
        pause
        exit /b 1
    )
)

REM uv 経由で実行（依存解決・キャッシュ活用）
echo [INFO] Starting training with overrides: %*
.venv\Scripts\uv run --active python -m src.main %*
```

### `setup_env.bat` 例

```bat
@echo off
echo [INFO] Setting up uv environment...
uv venv .venv --python 3.12
.venv\Scripts\uv pip install -e . --no-cache-dir
echo [INFO] Setup complete.
```

## 代替案と却下理由

| 代替案 | 却下理由 |
|---|---|
| PowerShell スクリプト (`.ps1`) | 実行ポリシー制限・署名必要・ユーザー敷居高 |
| `conda` 環境 | 重い・起動遅い・uv と互換性薄 |
| `pip` + `venv` 手動 | 再現性低い・キャッシュ管理手動・ロックファイルなし |
| `just` / `task` 等タスクランナー | 追加依存・学習コスト・Windows標準でない |

## 結果

### 正の影響
- **ワンコマンド実行**: `train.bat` ダブルクリック or `train.bat +max_steps=100`
- **環境自動修復**: `.venv` なければ `setup_env.bat` 自動実行
- **uv 高速キャッシュ**: 2回目以降 `uv pip install` 数秒で完了
- **Hydra オーバーライド透過**: `%*` でそのまま渡せる
- **CI/CD 移植容易**: `.bat` → `.sh` 置換で Linux 対応可能

### 負の影響
- **Windows限定**: Linux/macOS では `.sh` 別途必要
  - 対処: 将来的に `train.sh` 同等品追加予定
- **uv バージョン固定化**: `uv.lock` 管理推奨（現状 `pyproject.toml` のみ）
  - 対処: `uv lock` 採用検討中

## 検証

- `train.bat +max_steps=1` で学習起動〜1ステップ完了確認済み
- 環境削除後 `train.bat` で自動再構築〜起動確認済み