# ADR-0039-wsl2-linux-production-migration: 本番学習環境の WSL2 (Linux) への移行意思決定

- **Status:** Accepted
- **Date:** 2026-07-18
- **Deciders:** Ayato-labs (ayato-labs)

## Context

当初、本プロジェクトは Windows ネイティブ環境での互換性を維持しながら開発を進めてきた（ADR-008、ADR-032など）。
しかし、データセットの大規模化にともない、以下の要因から Windows ネイティブ環境での学習速度の低下およびイテレーション効率の悪化が無視できないレベルに達した：

1. **最適化機能の Windows 上での利用不可**:
   - `torch.compile`（PyTorch Inductor コンパイル）は Windows 環境では Triton バックエンドが利用できず非常に不安定であるため、無効化を余儀なくされていた。
   - `liger-kernel`（Triton製 RMSNorm / SwiGLU / CrossEntropy）が Windows で動作せず、VRAM削減の恩恵（バッチサイズ拡大）を受けられなかった。
   - `flash_attention_2` も Windows 環境ではビルドや利用が困難であった。
2. **学習時間の爆発**:
   - データ量が約2.28GBから約4.34GBに拡大したことで、3エポック時のステップ数は約10万ステップに達し、Windows上での想定学習時間は **1,127時間（約47日）** という現実的ではない規模となった。

学習時間短縮による開発イテレーション速度向上のため、本番学習環境を Linux（WSL2）へ移行し、これらすべての高速化機能を解放する必要性が生じた。

## Decision

本プロジェクトにおける「本番フル学習」の公式推奨環境を、**Windows ネイティブから WSL2 (Ubuntu 22.04+) に完全移行**し、Linux 用の最適化機能（`torch.compile`, `use_liger_kernel`, `flash_attention_2`）を前提とした設計に変更する。

### 変更および移行の基本ルール：
1. **WSL2内の高速ファイルシステムの利用**: I/Oボトルネックを防ぐため、プロジェクトファイルは `/mnt/c/...` ではなく、WSL2 の ext4 領域（`~/`）に配置する。
2. **ストレージ重複の回避**: 大容量の `dataset.jsonl` は Linux 側へ物理コピーせず、Windows（`/mnt/c/...`）の実体ファイルを指すシンボリックリンクを作成して参照する。
3. **環境変数 TMPDIR の調整**: `/tmp`（WSL2のメモリ制限付き tmpfs）溢れによる OSError 28 回避のため、一時ファイルディレクトリをディスク上（`models/output/tmp`）に指定する。

## Consequences

### Pros
- **劇的な学習時間の短縮**: 1エポック制限への方針変更（`概念的要件定義書` 更新）と組み合わせることで、総学習時間は **1,127時間（約47日）から約141〜188時間（約6〜8日）**へと短縮されると想定される。
- **機能の全解放**: `torch_compile`, `use_liger_kernel`, `flash_attention_2` を安定して有効化できる。
- **VRAM効率の最大化**: Liger Kernel によるメモリ削減のおかげでバッチサイズを拡大でき、さらにステップ効率を向上できる。

### Cons
- **WSL2環境の初期構築コスト**: ユーザー側での WSL2 の有効化や Ubuntu セットアップの手間が1度だけ発生する。
- **OSの抽象化レイヤーオーバーヘッド**: WSL2 の GPU パススルーによる数%程度の仮想化オーバーヘッド、および Windows側とLinux側でファイルを別個に管理するオーバーヘッドが発生する。
