# ADR-031: ローカル tokenizer.json 直接読み込みへの移行

**日付**: 2026-07-12  
**ステータス**: Accepted  
**決定者**: Solo Developer

## コンテキスト

旧実装では `AutoTokenizer.from_pretrained("data/tokenizer.json")` を呼び出していたが、Hugging Face `transformers` ライブラリはこれを **ローカルパスとして認識せず、HF Hub上のリポジトリIDとして解釈** しようとする。

結果：
- `RepositoryNotFoundError` で失敗
- 認証トークン要求・レートリミット・ネットワーク必須
- オフライン環境で動作不可

## 決定

**`tokenizers` ライブラリの低級APIで直接読み込み、HF `AutoTokenizer` インターフェースに注入** する。

```python
# src/main.py
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer
from pathlib import Path

tokenizer_path = Path(config["tokenizer_path"])
if tokenizer_path.suffix == ".json" and tokenizer_path.exists():
    hf_tokenizer = HFTokenizer.from_file(str(tokenizer_path))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # ダミーでインターフェース取得
    tokenizer._tokenizer = hf_tokenizer
else:
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
```

## 代替案と却下理由

| 代替案 | 却下理由 |
|---|---|
| `tokenizer_config.json` + `vocab.json` + `merges.txt` 旧形式で配置 | SentencePiece BPE の `tokenizer.json` 単一ファイルが標準。変換コスト無駄 |
| HF Hub にプライベートリポジトリとしてアップロード | 認証管理・ネットワーク依存・CI複雑化。ローカル完結が原則 |
| `transformers` の `tokenizer_file` 引数使用 | 存在しない / 非公式API |

## 結果

### 正の影響
- **完全オフライン動作**: ネットワーク不要、認証不要
- **高速起動**: Hub API呼び出し・キャッシュ確認・ダウンロードなし
- **バージョン固定**: `tokenizer.json` がソースコードと共にGit管理されるため再現性確保

### 負の影響
- `AutoTokenizer` の一部メタデータ（`chat_template` 等）が欠落する可能性
  - 対処: 必要なら手動で `tokenizer.chat_template = "..."` 設定
- `gpt2` ダミートークナイザーの語彙サイズ(50257)と実語彙(64000)が異なる
  - 対処: `model.resize_token_embeddings(len(tokenizer))` で即座に合わせるため無害

## 検証

- `max_steps=1` ドライランで tokenizer 読み込み〜モデル埋め込みリサイズまで正常完了確認済み