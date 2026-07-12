# ADR-024: Windows Native Compatibility and DLL Load Order

## Status
Accepted

## Context
Windows環境での実行時に、`0xC0000005` (Access Violation / セグメンテーションフォールト) によるプロセスの強制終了が発生した。
調査の結果、`pyarrow` (arrow.dll) と `torch` などのネイティブライブラリ間での DLL ロード順序の競合が原因であることが判明した。
また、`deepspeed`, `bitsandbytes`, `flash-attn` などのライブラリが Windows 上で不安定、あるいは非互換であり、環境構築の大きな妨げとなっていた。

## Decision
Windowsネイティブ環境での安定した動作を最優先し、以下の措置を講じる。

1. **Linux専用ライブラリの除外**:
   - `deepspeed`, `bitsandbytes`, `flash-attn` を `pyproject.toml` の必須依存関係から削除する。
   - これらの機能は、インストールされている場合のみ利用する「オプショナルな機能」として扱い、未インストール時でもクラッシュせずに動作するよう実装する。

2. **インポート順序の固定**:
   - `src/main.py` のエントリーポイントの最上部で、`torch`, `transformers`, `datasets`, `mlflow` を明示的に先にインポートする。
   - これにより、最も基盤となる重量級のネイティブDLLを最初にメモリにロードさせ、後続のライブラリによる DLL 競合（メモリアドレスの不整合）を回避する。

3. **パッケージ管理の最適化**:
   - `uv` による決定論的な環境構築を採用し、依存関係の不整合を最小限に抑える。

## Consequences
### Pros
- Windows環境において `0xC0000005` クラッシュが発生しなくなり、安定して学習・検証が実行可能になった。
- 環境構築の手順が簡略化され、`.venv` の再構築時間が短縮された。
- Windows/Linux間での `pyproject.toml` の共通性が高まり、管理コストが低下した。

### Cons
- Windows環境では DeepSpeed (ZeRO) や Flash Attention による高度なメモリ最適化・高速化が利用できない。
- ただし、単一GPUの個人開発環境においては、PyTorch標準の `gradient_checkpointing` 等で十分なメモリ確保が可能であり、許容範囲内であると判断した。
