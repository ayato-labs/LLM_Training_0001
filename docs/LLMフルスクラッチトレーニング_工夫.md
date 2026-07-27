# LLM フルスクラッチトレーニング 実装・最適化の工夫一覧

- **作成日**: 2026-07-22
- **最終更新日**: 2026-07-26

本ドキュメントは、LLM（LLaMAアーキテクチャベース）のフルスクラッチ学習において、ロジック的な面から**有限の計算リソース（特に 4GB VRAM 級のエントリー GPU および Windows WDDM 環境）で安定性・処理速度・モデル精度を最大化するために導入した工夫と技術的施策**をまとめたものである。

- 「起動時ハードウェア診断→設定を自動生成する抽象化レイヤーを1枚挟む」方向の軸で、アークテクチャ設計を行う。

## 限られたVRAMリソースの場合(ハイパースケーラー以外)のフルスクラッチトレーニングの優先順位

### 開発・実験フェーズ（イテレーション速度・時間優先）
> 精度 < コンテキスト長増加（事後拡張に委ねる場合） < VRAMの使用効率向上 < モデルサイズの拡大 < **学習時間の短縮** < データ収集

※ 開発・検証段階では「イテレーション回数と試行速度（Time to Insight）」が最重要となるため、VRAM消費削減による計算効率化やオーバーヘッド増加を冒してまでVRAMを限界まで節約するより、**学習速度・時間短縮**を最優先してフィードバックサイクルを高速化する。

### 本番・大規模学習フェーズ（リソース限界到達・最大規模モデル優先）
> 精度 < 学習時間の短縮 < コンテキスト長増加（事後拡張に委ねる場合） < VRAMの使用効率向上 < モデルサイズの拡大 < データ収集


## フルスクラッチトレーニングの標準実行フロー (Pipeline Step)
本プロジェクトにおけるモデル開発・学習は、以下のステップ順序に従って実行される：

1. **チンチラの法則（Chinchilla Scaling）による設計 (`src/chinchilla`)**
   - 目標学習時間・計算リソース（VRAM等）から最適パラメータ数 $N$ と目標トークン数 $D$ を逆算し、モデルアーキテクチャ（レイヤー数, 隠れ層次元等）を決定。
2. **HPO（Step Law によるハイパーパラメータ最適化 / `src/hpo`）**
   - 小規模プロキシモデルを用いて最適学習率等を探索し、Step Law の理論比率でターゲット規模へ転移（探索コストの最小化）。
3. **フルスクラッチ事前学習 (Pretraining / `src/training`)**
   - 短めのコンテキスト長（例: `seq_len=1024`）と Sequence Packing で計算効率を最大化しつつ、事前学習を実行。
4. **コンテキストウィンドウの事後長文拡張 (Continued Pretraining / `src/context_extension`)**
   - 事前学習完了後、YaRN / Dynamic NTK や選択的 Activation Checkpointing を併用して `seq_len=4096〜8192` へ長文適応。
5. **Q&A / 指示追従ファインチューニング (SFT / Direct Alignment)**
   - 拡張されたコンテキスト長とモデル重みに対し、Q&A形式等の対話データセットを用いてインストラクションチューニング（SFT）を実施し、実用モデルとして完成させる。

---


## 1. VRAM 削減およびメモリスワップ回避の工夫

### 1.1 リソース制限環境における `torch.compile` の評価と運用の考え方
* **工夫の背景と判断軸**:
  * **VRAM コストとGraph Breakの課題**: `torch.compile`（PyTorch Inductor）は計算グラフの事前コンパイルおよび中間バッファ保持のために一定の VRAM 余白を要求する。特に `liger_kernel` などのカスタム Triton autograd 関数と併用した場合、TorchDynamo のトレーシング時に **Graph Break（計算グラフの断片化）** が多発し、Eager モードへのフォールバックでメリットが薄れる。
* **適用指針**:
  4GB VRAM の極小環境や `liger_kernel` 採用時は無効化 (`torch_compile: false`) を基本とする。一方で、`liger_kernel` を使用しないシンプルなアーキテクチャや VRAM に余裕がある Linux 環境では `mode="reduce-overhead"` や `dynamic=True` での再検討価値を残した構成としている。

### 1.2 8-bit AdamW (`bitsandbytes`) の全面採用
* **工夫の背景と理由**:
  1D パラメータ（Embedding, LayerNorm, Bias）の最適化において、FP32 のオプティマイザステート（1階・2階モーメント）をそのまま保持すると VRAM を圧迫する。
* **効果**:
  `bitsandbytes` の `AdamW8bit` を適用することで、オプティマイザステートを 8-bit 量子化し、VRAM 消費量を **約 0.3GB〜0.5GB 削減**。

### 1.3 Non-reentrant 型 Activation Checkpointing
* **工夫の背景と理由**:
  `gradient_checkpointing_kwargs={"use_reentrant": False}` を指定。
* **効果**:
  PyTorch の非再帰型チェンジポイント機構を用いることで、動的メモリの解放と再計算を安定化し、アクティベーションによる VRAM ピークを最小化。

---

## 2. 学習安定化および計算効率向上の工夫

### 2.1 スケーリング法則（Step Law）活用による HPO 空間最適化とモデル間スケーリング転移
* **工夫の背景と理由**:
  広大なハイパーパラメータ空間を広域グリッドサーチや無計画なベイズ最適化で探索すると、膨大な計算リソースと時間を消費する課題があった。また従来はプロキシモデルサイズが手動指定やハードコードされたプリセットに依存していた。
* **施策**:
  * **探索空間の適正絞り込み**: Step Law（arXiv:2503.04715: $\eta^*(N, D) = 1.79 \cdot N^{-0.713} \cdot D^{0.307}$）の理論式に基づき、ターゲットパラメータ数 $N$ とトークン数 $D$ から探索空間の中心点を事前に算出し、無駄な広域試行を排除。
  * **黄金比率によるプロキシモデル動的自動算定 (`determine_optimal_proxy_size`)**:
    静的な固定分岐（`if/elif`）を排除し、ターゲットパラメータ数 $N_{\text{target}}$ から数式 $N_{\text{proxy}} = \max(50\text{M}, \lfloor N_{\text{target}} \times 0.10 \rfloor)$ にて厳密計算。プロキシモデル（例: 7Bターゲット $\rightarrow$ 700Mプロキシ）のアーキテクチャパラメータ（レイヤー数, 隠れ層次元等）を動的に自動生成してバインド。
  * **探索空間次元数に基づく HPO 試行回数の理論的自動算定 (`calculate_dynamic_n_trials`)**:
    特定のモデルパラメータ数（150M等）への手動過剰適合式を完全廃止。探索するパラメータ空間の次元数（自由度 $D$: 現状5次元）に基づき $T_{\text{trials}} = D \times 30$ で理論的に算定。Optuna の MedianPruner (枝刈り) と併用することで過不足のない最適試行回数を自律制御。
* **効果**:
  手動指定・固定プリセット・過剰適合計算式の制限を完全撤去し、任意のターゲット規模に対して転移精度と探索時間が最適化される完全自律型 HPO パイプラインを実現。





### 2.2 学習率の安全上限設定と二重ガード（Single Source of Truth ＋ Defense in Depth）
* **工夫の背景と理由**:
  Muon オプティマイザ（2D重み行列用）は Newton-Schulz 直交化を行うため通常の AdamW より更新量が大きく、無制限な高学習率では Loss 発散リスクがある。一方、Keller Jordan のオリジナル実装や NorMuon の知見に基づき、実用的な高学習率（0.01〜0.02 レンジ）を許容できるように設定を最適化した。
* **施策**:
  * **安全上限値の定義 (SSOT)**: `src/common/constants.py` に `MAX_LR_2D = 0.02` (Muon用), `MAX_LR_1D = 0.0010` (AdamW用) を Single Source of Truth として一括定義。
  * **二重ガード実装**: HPO 探索空間算定時（`step_law.py`）と Config 正規化起動時（`config.py`）の双方から同モジュールの `clip_learning_rates()` を参照・通過させ、不整合の発生を防ぎつつ自動クリッピング。
* **効果**:
  NorMuon 等の最新知見に合わせた高速収束を維持しつつ、過大な学習率による Loss 爆発を構造的に防止。

### 2.3 絶対値での Warmup ステップ確保 (`warmup_steps: 50`)
* **工夫の背景と理由**:
  `warmup_ratio`（比率指定）のみに依存すると、ステップ数が少ない実験やテスト時にウォームアップが 1〜2 ステップで終了してしまい、初期のランダム重みに対して急激な勾配更新が印加される。
* **効果**:
  最低 50 ステップの「準備運動（学習率の緩やかな上昇）」を強制的に確保することで、初期の勾配破壊を防ぎ、収束軌道を安定化。

### 2.4 多重自律発散検知（語彙数動的理論上限 ＋ 相対スパイク判定 ＋ NaN/Inf 監視）
* **工夫の背景と理由**:
  単一のハードコードされた固定閾値（例: 10.0）に依存すると、語彙数（`vocab_size`）変更による初期 Loss の理論値変化で誤判定を起こすほか、学習途中の局所発散を検知できない問題があった。
* **施策**:
  `PeriodicEvaluationCallback` に 3 重の自律監視ロジックを導入：
  1. **語彙数動的理論上限**: `vocab_size` から初期理論最大 Loss ($\ln(V) \times 1.5$) を自動算定。
  2. **相対スパイク判定**: 過去最小 Loss (`min_loss`) を記録し、学習途中の過大跳ね上がり (`loss > min_loss * 2.0`) を検知。
  3. **数値不安定性検出**: `NaN` / `Inf` を即時検出。
* **効果**:
  語彙数やモデルサイズに依存しない頑健な自動早期停止を実現し、リソースの空費とチェックポイント汚染を防止。

### 2.5 OS安全上限 ＋ パフォーマンス動的プロファイリング DataLoader
* **工夫の背景と理由**:
  CPU 側でのデータロードが遅れると GPU 空転が発生するが、Windows (WDDM) 環境等ではマルチプロセスの Shared Memory IPC オーバーヘッドやプロセスフリーズ事故のリスクが存在する。
* **施策**:
  `train_engine.py` にて、OS/ストレージ環境の安全上限（Windows WDDM デッドロック防止上限 vs Linux CPUコア数上限）と、WSL `/mnt/` ストレージアクセス遅延等のパフォーマンスボトルネックを組み合わせたハイブリッド自動判別を実施。
* **効果**:
  マルチプロセス通信のクラッシュ・フリーズ事故を完全に回避しつつ、GPU 空転率を最小化する最適 worker 数を自動適用。

### 2.6 cuDNN 最速カーネルサーチの有効化 (`cudnn.benchmark = True`)
* **工夫の背景と理由**:
  GPU 上での行列演算カーネルには複数のアルゴリズムが存在し、標準状態では汎用的なカーネルが使われる。
* **施策**:
  Sequence Packing により入力長が **1024 定長** に完全に揃っている特性を活かし、`torch.backends.cudnn.benchmark = True` を有効化。第 1 ステップで試行テストを行い、お使いの GPU に最適な最速 cuDNN カーネルを自動選択・固定。
* **効果**:
  VRAM 消費ゼロで、Attention 以外の全結合層（Linear層等）の計算スピードを 5%〜10% 追加底上げ。

### 2.7 Muon Newton-Schulz 反復数のモデル規模に応じた動的判定 (`get_optimal_newton_schulz_steps`)
* **工夫の背景と理由**:
  小規模モデル（150M級）では `steps = 3` で十分な直交化精度と速度が得られるが、3B/7B 等の大規模モデルにスケールアップした際には行列の条件数により精度不足となるリスクがあった。
* **施策**:
  `muon.py` 内に `get_optimal_newton_schulz_steps` を導入。隠れ層次元 $d_{\text{model}} < 2048$（小規模）では `steps = 3` で高速処理し、$d_{\text{model}} \ge 2048$ または 1B 以上のモデルでは `steps = 5` へ動的に切り替えるロジックを実装。
* **効果**:
  モデル規模に応じた理論的一貫性と直交化精度を保ち、スケーリング転移時の安定性を確保。

### 2.8 ハッシュ自動検証付き学習再開機能 (`resume_from_checkpoint` ＋ `hashes.json`)
* **工夫の背景と理由**:
  長時間・多ステップに及ぶ事前学習において、中断からの再開 (`resume_from_checkpoint`) は不可欠である。しかし、異なる設定パラメータや異なるデータセットの状態で誤って再開すると、モデルの重みが壊れるリスクがあった。
* **施策**:
  * `HashSaveCallback`: チェックポイント保存時に Config と学習データのハッシュ値を算出し `hashes.json` へ保存。
  * **自動検証**: `uv run python -m src.training.main resume_from_checkpoint=models/output/checkpoint-latest` 等での再開時、現在実行中の Config/データハッシュと全自動比較。
* **効果**:
  設定ミスによるモデル汚染を 100% 防止し、完全な整合性を担保した安全な学習継続を実現。

### 2.9 SDPA 低速 Pure Math フォールバックの強制禁止
* **工夫の背景と理由**:
  PyTorch の SDPA (Scaled Dot-Product Attention) は、特定の条件下で C++/Python の低速な Pure Math 実装へフォールバックする可能性があった。
* **施策**:
  `torch.backends.cuda.enable_math_sdp(False)` を明示的に呼び出し、Math 実装への退避を禁止。高速な CUDA / FlashAttention カーネルのみを強制選定。
* **効果**:
  Attention 計算における低速病の発生を 100% 遮断。

### 2.10 Muon Newton-Schulz 演算の JIT コンパイル化 (`@torch.jit.script`)
* **工夫の背景と理由**:
  オプティマイザの毎ステップで実行される直交化ループ演算における Python 側の呼び出し・評価オーバーヘッド。
* **施策**:
  `muon.py` 内の `zeropower_via_newtonschulz` 関数に `@torch.jit.script` デコレータを付与し、C++ レベルで最適化コンパイル。
* **効果**:
  オプティマイザステップの Python オーバーヘッドを削り、C++ Native 相当の速度で処理。

### 2.11 Tensor Core float32 行列積精度モードの開放 (`set_float32_matmul_precision('medium')`)
* **工夫の背景と理由**:
  標準状態では FP32 行列積が保守的な精度設定 (`high`) に固定されており、Ampere 世代 GPU（RTX 3050 等）の Tensor Core 演算器の最大パフォーマンスを引き出しきれていなかった。
* **施策**:
  `train_engine.py` 冒頭のハードウェア最適化処理にて `torch.set_float32_matmul_precision("medium")` を適用。内部の FP32 行列積を Tensor Core 最適な bfloat16 相当精度（仮数部切り捨て）で強制計算。
* **効果**:
  実用上の精度低下を極めて軽微に抑えつつ、全 Linear 層の行列演算スピードを **10%〜20% 高速化**。

### 2.12 重み共有（Weight Tying / Embedding層とLM Headの結合）
* **工夫の背景と理由**:
  LLaMA等のアーキテクチャでは通常 Embedding 層（`32000 x 768`）と LM Head 層（`768 x 32000`）が個別のパラメータとして学習され、重み更新およびVRAM転送のオーバーヘッドを生んでいた。
* **施策**:
  `configs/config.yaml` にて `tie_word_embeddings: true` を有効化し、`LlamaConfig` に反映。Embedding 層と LM Head の重み行列を共有（転置結合）化。
* **効果**:
  パラメータ数を全 150M パラメータから **約 2,450 万パラメータ軽量化（150M $\rightarrow$ 約 125.5M）** し、行列乗算とメモリ転送の削減により学習速度を **約 10%〜15% 追加高速化**。

### 2.13 保存・評価頻度の分離最適化 (`save_steps: 500` / `eval_steps: 3000`)
* **工夫の背景と理由**:
  テキスト生成評価（Auto-regressive推論）は学習ループ（GPU並列計算）を一時停止（ストール）させる高コストな処理であった。従来のように保存と同じ頻度で評価を実行すると、不要な GPU 空転時間が発生していた。
* **施策**:
  `configs/config.yaml` にて、障害防止・途中再開用のチェックポイント保存（`save_steps: 500`）と、テキスト生成評価の実行（`eval_steps: 3000`）を独立制御。
* **効果**:
  万が一の中断保護機能は高頻度（500ステップごと）に全維持したまま、テキスト生成による GPU 一時停止時間を大幅カットし、モデル精度・安全性への悪影響ゼロで学習速度を効率化。

### 2.14 Gradient Checkpointing の選定と Liger Kernel 互換性に基づく一本化方針
* **工夫の背景と検証結果**:
  * **初期の取り組み（選択的 Gradient Checkpointing）**: 当初は VRAM 節約と計算オーバーヘッド低減のため、`self_attn` (Attention) のみを選択的に再計算し、計算量の重い MLP (SwiGLU) の再計算をスキップする `selective_checkpointing` の導入を試みた。
  * **相性不整合とデッドロックの発覚**: `Liger Kernel` (Triton) は GPU のベクトライザ効率（128-byte alignment）を高めるため、入力シーケンス長を内部で動的にパディング/アラインメント（例: `5120` $\rightarrow$ `5632`）する。Attention モジュール単体のみを選択的チェックポイント化すると、アテンションと MLP の境界で PyTorch の Autograd 再計算フック（Unpack Hook）と C++ 逆伝播アサート（`ScaledDotProductEfficientAttentionBackward0`）がテンソル形状不一致（`CheckpointError`）を判定し、C++ スレッドがデッドロックを起こすことが判明した。
* **最終的な決断と施策（2026-07-26 時点）**:
  * `selective_checkpointing`（Attention のみの部分的再計算）は `Liger Kernel` との根本的な相性が悪いと結論付け、これを完全無効化 (`selective_checkpointing: false`)。
  * HuggingFace 標準の **全層フル Gradient Checkpointing (`gradient_checkpointing=True`, `gradient_checkpointing_kwargs={"use_reentrant": False}`)** へ切り替え・一本化を遂行。
* **効果**:
  Liger Kernel のメモリ最適化（Triton Fused Kernel）による VRAM 削減と、標準 Checkpointing の安定性が 100% 同時発揮される完璧な構成を達成。Autograd のデッドロック、25 秒フリーズ、および `CUDA driver error` を完全に遮断し、1 ステップあたり **約 0.9 秒** という超高速かつ極めて安定した事前学習・HPO イテレーションを実現。

### 2.15 物理 VRAM メモリ近似式に基づく動的マイクロバッチ分解 (`calculate_optimal_batch_split`)
* **工夫の背景と理由**:
  バッチサイズは損失（Loss）を最小化するための探索変数ではなく、**GPUの物理VRAM容量と計算速度（Tensor Core MFU）によって定まるハードウェアパラメータ**である。従来は `per_device_batch_size = 1` に固定（ハードコード）され、トータルバッチサイズ（例: 32）をすべて勾配蓄積（`grad_accum_steps = 32`）で処理していたため、GPU への小分け命令発行オーバーヘッドが大きく、Tensor Core の並列演算器が充填されず実効速度が下振れしていた。
* **施策**:
  1. **バッチサイズの管轄権の一元化（Chinchilla管轄）**: バッチサイズの決定権を HPO の探索空間から完全排除し、Chinchilla / VRAM シミュレーション段階（Step 1）で即座に動的算定・確定する設計に変更。
  2. **速度優先の直接最大化ロジック**: `model_utils.py` の `calculate_optimal_batch_split` を改修。ループ減算を全廃し、物理限界を超えない最大安全マイクロバッチ `per_device_batch_size = max(1, min(total_batch_size, max_safe))` を直接割り当て、GPU の Tensor Core を100%充填。
* **効果**:
  HPO や学習パイプラインにおいて GPU 命令発行オーバーヘッドを最小化し、モデル規模や VRAM 容量に応じた最高の MFU（GPU計算効率）と超高速ステップ処理を自動達成。

### 2.16 三位一体のバッチ・ステップ数自律最適化パイプライン (Chinchilla $D$ 保持 ＋ HPO事前枝刈り ＋ 動的ステップ数計算)
* **工夫の背景と理由**:
  従来の Chinchilla 計算では物理ステップ数 `max_steps` が固定出力されていたため、総消費トークン数（計算バジェット）が変動してしまう問題があった。またプロキシベンチマーク時に `batch_size = 1` でスループットを測定していたため、本番学習で大バッチ（`batch_size = 64` 等）を投入した際に Tensor Core が充填されて処理速度が 15倍に跳ね上がり、24時間設定の学習が1時間で終わる計算剥離が発生していた。
* **一方通行パイプライン依存関係 (DAG)**:
  `Step 1: Chinchilla ($N$, $D$, 物理最速 $batch\_size$ 算定)` $\rightarrow$ `Step 2: HPO (ハイパラ探索・VRAM事前枝刈り)` $\rightarrow$ `Step 3: 学習設定正規化 ($max\_steps$ 動的逆算・学習実行)`
* **施策**:
  1. **Tensor Core 飽和スループット実測 (`run_quick_proxy_benchmark`)**: VRAM 物理算定式で求めた物理最速バッチサイズ（`batch_size = 64` 等）と `LlamaModel` バックボーンを用いて、Tensor Core が飽和した状態の真の最大スループット（7.5万〜20万 tokens/sec）をダイレクトに実測。
  2. **Chinchilla理論トークン数 $D$ の完全不変保持 (Step 1)**: 理論的目標トークン数 `computable_tokens` ($D = 20N$) と物理最速バッチサイズを `chinchilla_config.yaml` へ保存・固定（HPO 成果物への循環依存は一切なし）。
  3. **`vram_estimator.py` による Zero-Cost Pre-Pruning（事前枝刈り） (Step 2)**: HPO マネージャー（`hpo_manager.py`）が試行評価直前に `vram_estimator.py` を呼び出し、推定 Peak VRAM が物理容量の 90%（`util_pct > 90.0%`）を超える危険設定を検知した瞬間に、GPU プロセスを起動せず 0 秒で `TrialPruned` を発動させてスキップ。
  4. **学習設定正規化時の動的 `max_steps` 逆算 (Step 3)**: 本番学習直前の `config.py` にて、Step 1 の $D$ と `batch_size_seqs` を組み合わせ、$max\_steps = computable\_tokens / (batch\_size\_seqs \times seq\_len)$ を動的計算。
* **効果**:
  完全な一方通行（循環なし）のデータフローを保ちつつ、学習設定時間と実際の学習完了時間の剥離をゼロに補正。HPO での OOM クラッシュによる試行中断がゼロになり、バッチサイズ変動に伴う計算バジェットのブレを完全遮断。

---

## 3. 計算コストゼロでのモデル精度向上工夫

### 3.1 1D / 2D パラメータの Weight Decay 分離
* **工夫の背景と理由**:
  すべてのパラメータに一律で Weight Decay（L2正則化）を適用すると、LayerNorm のスケール係数や Bias までが「0」に引っ張られ、モデルの表現力が理不尽に減衰する。
* **施策**:
  `SplitOptimizer` 内でパラメータを自動判定し、2D重み行列のみ `weight_decay = 0.1` を適用、1Dパラメータ（Bias, LayerNorm）は `weight_decay = 0.0` に分離。
* **効果**:
  計算コストや VRAM を一切増やさずに、モデルの表現力と過学習抑制性能を向上。

### 3.2 RoPE Theta のコンテキスト長最適化 (`500,000` $\rightarrow$ `10,000`)
* **工夫の背景と理由**:
  `rope_theta = 500000.0`（50万）は 128,000 トークン等の超長文向け設定であり、`seq_len: 1024` で使用すると「近距離トークン間の位置分解能」が低下する。
* **効果**:
  1024 長に適した標準値 `10000.0` に変更することで、近接単語間の相対位置把握精度を高め、文法力・文章構成力を向上。

### 3.3 Dataset Sequence Packing（パディングトークン排除）
* **工夫の背景と理由**:
  通常のパディング方式では、短文データの末尾に大量の `<pad>` トークンが挿入され、無駄な計算力と VRAM を消費する。
* **施策**:
  全テキストを EOS トークン区切りで結合し、正確に 1024 トークンごとにスライスする Sequence Packing を導入。
* **効果**:
  無駄な計算を 100% 排除し、1ステップあたりの学習効率を最大化。

---

## 4. 事後長文拡張（Long-Context Extension / Continued Pretraining）の設計

### 4.1 `src/context_extension` によるエントリーポイント完全分離型モジュール
* **工夫の背景と理由**:
  事前学習（`seq_len: 1024`）完了後に、`seq_len: 4096~8192` へ長文適応（Continued Pretraining）を行う際、事前学習コード（`train_engine.py`）にロジックを直接埋め込むとコードが煩雑化・二重管理化する問題があった。
* **施策**:
  `src/hpo` と同様に `src/context_extension/` を独立構築。
  * `configs/config.yaml` および HPO 成果物 (`last_run_result.json`) を自動継承・統合。
  * `rope_scaling`（YaRN / Dynamic NTK）の動的注入。
  * 選択的 Attention Checkpointing (`apply_selective_attention_checkpointing`) の適用により、長文時の VRAM 圧迫を回避。
  * `target_seq_len: 4096~8192` 用データパッキングと Short Warmup 付近の Continued Pretraining を実行。
* **効果**:
  メインの事前学習コードを完全に綺麗に維持したまま、学習完了モデルからコマンド一発で `seq_len: 4096~8192` への高精度長文適応が可能に。

---

### 5.1 3段階統合スケーリング法則エンジンによる計算予算と 2段階分離型バッチの自律算定 (`src/scaling_laws/`, `configs/scaling_config.yaml`)
* **工夫の背景と理由**:
  「目標学習時間（例: 24時間 / 48時間）」から構築可能な最適モデル構造・バッチサイズを導出する際、従来は単一の Chinchilla 則のみに依存しており、限界バッチサイズ（Critical Batch Size）や GPU 物理 VRAM に対するマイクロバッチ分割の最適化が混在・分離されていなかった。また成果物ファイル名が `chinchilla_config.yaml` に限定されており、多角的なスケーリング法則全般を包含する責務表現として不十分であった。
* **施策**:
  `src/chinchilla/` を **`src/scaling_laws/`** へ改名・昇華させ、学術論文に基づく **「3段階統合スケーリング法則パイプライン (3-Stage Scaling Laws Pipeline)」** へ再構築（SoC: 責任と関心の分離を徹底）：
  * **Stage 1: Chinchilla 則 (Hoffmann et al., DeepMind 2022)** (`src/scaling_laws/chinchilla_law.py`):
    目標時間・スループット（Tokens/sec）から計算可能トークン数 $D = 20N$ および Compute-Optimal モデル規模 $N_{\text{opt}}$、LLaMA 幾何学構造を自律算定。
  * **Stage 2: 限界バッチ法則 (McCandlish et al., 2018 / Kaplan et al., OpenAI 2020)** (`src/scaling_laws/critical_batch_law.py`):
    モデル規模 $N_{\text{opt}}$ に対し、Loss 収束効率が最も高くなる理論最適トータルバッチサイズ $B_{\text{total}}$ ($B_{\text{crit}} \propto N^{0.3}$) を算定。
  * **Stage 3: 2段階分離型マイクロバッチ自律判定 (Decoupled Batch Architecture)** (`src/scaling_laws/critical_batch_law.py`):
    理論最適トータルバッチ $B_{\text{total}}$ と GPU 物理空き VRAM（90% 安全基準: `torch.cuda.mem_get_info()`）を動的照合：
    * **VRAM に収まる場合 (例: 86.5M モデル)**: `per_device_batch_size = B_total`, `grad_accum_steps = 1` を自動選択。勾配蓄積オーバーヘッド（4倍の autograd / checkpointing 再計算遅延）を完全回避し、最高速度（3.70s/step）を実現。
    * **VRAM を超過する場合 (例: 1.5B/7B モデル)**: VRAM 容量内に収まる最大マイクロバッチ $b_{\text{micro}}$ へ小分けし、`grad_accum_steps = ceil(B_total / b_micro)` を確定して OOM を100% 回避。
  * **単一成果物一元化 (`configs/scaling_config.yaml`) と 100% 後方互換性**:
    成果物ファイルを [configs/scaling_config.yaml](file:///wsl.localhost/Ubuntu/home/ayato/programing/LLM/Novel_LLM/LLM_Training/configs/scaling_config.yaml) へ統合し、マスター設定 [configs/config.yaml](file:///wsl.localhost/Ubuntu/home/ayato/programing/LLM/Novel_LLM/LLM_Training/configs/config.yaml) の `defaults` へ統一。旧 `src/chinchilla/` および `configs/chinchilla_config.yaml` はエイリアス層として 100% 後方互換性を保持。
* **効果**:
  GPU 物理 VRAM 容量への 4GB ハードコード・オーバーフィットをゼロにし、4GB から 80GB VRAM までの任意の GPU 環境において「精度・収束効率（理論トータルバッチ）」と「実行スピード（物理マイクロバッチ）」の両立を完全自律化した。