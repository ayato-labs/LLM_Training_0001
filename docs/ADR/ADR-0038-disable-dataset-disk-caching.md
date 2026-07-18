# ADR-0038-disable-dataset-disk-caching: Hugging Face datasets ディスクキャッシュの無効化

- **Status:** Accepted
- **Date:** 2026-07-18
- **Deciders:** Ayato-labs (ayato-labs)

## Context

ローカルの JSONL データセットを用いた学習および HPO（ハイパーパラメータ最適化）の実行過程において、Cドライブのストレージ（Hugging Face datasets のデフォルトキャッシュディレクトリ）が 50GB 以上圧迫される問題が発生した。

このストレージ肥大化の主な原因は、Hugging Face `datasets` ライブラリの以下の挙動にある：
1. **処理・変換結果の自動永続化**: `datasets.map()` によるトークン化（Tokenize）やシーケンスパッキング等の処理を実行するたびに、変換後データが Arrow 形式等のキャッシュファイルとしてディスクに書き出される。
2. **キャッシュの累積**: スクリプトの修正や再実行のたびに、過去のキャッシュファイルが自動削除されずディスク上に残り続ける。
3. **オンメモリ動作との重複**: [ADR-0037](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/docs/ADR/ADR-0037-packing-and-microbatch-optimization.md) で定義されている通り、本プロジェクトのデータ処理パイプラインではロード・前処理後の全データセットを `ds.set_format(type="torch")` によりメモリ（RAM）上にロードして保持している。

したがって、一度メモリにロードした後はディスク上のキャッシュは一切参照されず、ディスクへのキャッシュ書き出しは無駄なストレージ消費と書き込みI/O負荷を発生させるのみであった。

## Decision

本プロジェクト全体の Hugging Face `datasets` ディスクキャッシュ機能を完全に無効化する。

### 変更点

1. **学習エンジン (`train_engine.py`)**:
   `from datasets import load_dataset` に加え、`from datasets import disable_caching` をインポートし、インポート直後（またはデータロード前）に `disable_caching()` を呼び出す。

2. **HPOスクリプト (`find_hparams.py`)**:
   同様に `from datasets import disable_caching` をインポートし、ファイル上部で `disable_caching()` を呼び出す。

3. **概念的要件定義書 (`概念的要件定義書.md`)**:
   「5.2 ストレージ制約」セクションに、自動キャッシュ機能を無効化している方針を追記し、ストレージ容量の最小化を憲章に加える。

## Consequences

### Pros
- **ストレージ容量の劇的な節約**: 数十GB単位で蓄積されていた不要な Arrow キャッシュファイルが一切生成されなくなる。
- **ディスク I/O の低減**: 一時ファイルの書き込みが発生しないため、ストレージへの負荷が減り、前処理立ち上がり時のオーバーヘッドも僅かに改善する可能性がある。
- **メモリ動作の維持**: データセットは引き続きメモリ（RAM）上で保持されるため、学習ループ（DataLoader）の速度低下やI/Oボトルネックは発生しない。

### Cons
- **キャッシュなしでの再起動**: 学習プロセスの再起動時に前処理（トークン化など）が毎回実行される。しかし、本プロジェクトのローカルデータセット規模と ThreadPoolTokenizer によるマルチスレッド処理（[ADR-0031](file:///c:/Users/saiha/My_Service/programing/LLM/Novel_LLM/LLM_Training/docs/ADR/ADR-0031-windows-threadpool-tokenization.md)）により、前処理自体は数秒〜数十秒で完了するため、実用上のデメリットは極めて低い。
