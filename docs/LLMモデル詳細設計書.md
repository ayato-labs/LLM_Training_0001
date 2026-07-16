# LLMモデル詳細設計書 (LLM Model Specification)

本ドキュメントは、小説執筆用LLMモデル（Llama系アーキテクチャ）の深層学習モデル構造およびその学習（最適化）ロジックに関する詳細設計書である。

---

## 1. モデルアーキテクチャ概要 (Model Architecture Overview)

本モデルは、**Llama**ベースの自己回帰型デコーダのみのトランスフォーマー（Autoregressive Decoder-Only Transformer）である。主言語モデルのパラメータ数と隠れ層次元はハードウェア制限（メモリ量およびVRAM）に合わせてスケール調整されている。

### モデル基本パラメータ構成 (150Mモデル基準)
*   **モデルタイプ**: LlamaForCausalLM
*   **総パラメータ数**: 約 150M (1億5000万パラメータ)
*   **隠れ層次元数 ($d_{\text{model}}$)**: 768
*   **レイヤー数 (層数)**: 12
*   **アテンションヘッド数 ($n_{\text{heads}}$)**: 12 (Query用)
*   **KVヘッド数 ($n_{\text{kv\_heads}}$)**: 3 (Key/Value用, GQAを採用)
*   **FFN中間層次元数 ($d_{\text{ffn}}$)**: 3072 (SwiGLU)
*   **語彙サイズ ($V$)**: 64,000 (SentencePiece BPE)
*   **最大コンテキスト長 ($T$)**: 1024 / 2048

---

## 2. ディープラーニング層別詳細設計 (Layer Details)

### 2.1 トークン埋め込み層 (Token Embedding Layer)
入力トークンIDの列を隠れ層の連続ベクトルに写像する。
$$X_0 = \text{Embedding}(W_{\text{emb}}, \mathbf{x}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
*   語彙サイズ $V = 64,000$ はアテンション計算効率最大化のために選定されており、各トークンIDは $d_{\text{model}} = 768$ 次元の空間へ埋め込まれる。
*   勾配の安定化のため、埋め込みパラメータ $W_{\text{emb}}$ は後述の1Dオプティマイザ（AdamW）で最適化される。

### 2.2 アテンション層: Grouped-Query Attention (GQA)
アテンション計算でのメモリ帯域圧迫を軽減するため、アテンションメカニズムに **Grouped-Query Attention (GQA)** を採用している。

*   **ヘッド比率 (Q:KV Ratio)**: Query用ヘッド12に対し、Key/Value用ヘッドは3つに削減されている（比率 4:1）。
*   複数（4つ）のQueryヘッドが1つのKVヘッドのペアを共有してアテンションを算出する。これにより、推論時および学習時のKVキャッシュサイズがマルチヘッドアテンション（MHA）の4分の1になり、高速化と省VRAM化が実現されている。

### 2.3 位置エンコーディング: Rotary Position Embedding (RoPE)
位置情報の埋め込みには、相対位置情報を捉えることに適した **Rotary Position Embedding (RoPE)** をアテンションのQueryおよびKeyに適用する。
*   アテンション投影行列後のベクトルに対し、複素平面上での回転行列を適用することで相対位置をエンコードする。
*   **RoPE Base ($\theta$)**: $10000.0$
*   文脈の長期依存関係を安定して学習できるよう設計されている。

### 2.4 フィードフォワード層: SwiGLU (Swish Gated Linear Unit)
MLP（Multi-Layer Perceptron）ブロックには、ゲート機構を持った非線形活性化関数である **SwiGLU** を採用している。
$$\text{SwiGLU}(X) = (\text{Swish}(X W_{\text{gate}}) \otimes X W_{\text{up}}) W_{\text{down}}$$
*   $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$, $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$ の3つの学習可能パラメータ行列で構成される。
*   中間層次元 $d_{\text{ffn}}$ は 3072 次元の設定を採用。

### 2.5 正規化レイヤー: RMSNorm
各トランスフォーマーブロックの入力前および出力前に、パラメータ不要で計算効率が高い **RMSNorm (Root Mean Square Normalization)** を適用する。
$$\text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d} \sum_{i=1}^d X_i^2 + \epsilon}} \odot \gamma$$
*   平均値の減算を排除し、二乗平均平方根のみでスケーリングを行う。
*   計算の数値的安定性のための微小定数 $\epsilon = 1.0 \times 10^{-6}$。

---

## 3. 学習・最適化ロジック (Training & Optimization Logic)

### 3.1 損失関数 (Loss Function)
言語モデルの最適化ターゲットとして、標準的な自己回帰クロスエントロピー損失（Autoregressive Cross-Entropy Loss）を用いる。
$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})$$
ターゲットトークンとして `<pad>` トークン（ID: 3）は無視（Mask）され、損失計算から除外される。

### 3.2 パラメータ分割とオプティマイザ設計 (Muon / AdamW Split Optimizer)
深層学習パラメータの構造（テンソルの次元数）に応じて、オプティマイザの役割を分割する。

| パラメータカテゴリ | 適用対象テンソル | 適用オプティマイザ |
| :--- | :--- | :--- |
| **2Dパラメータ (行列)** | 各種アテンション投影ウェイト、FFN投影ウェイト ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) | **Muon** |
| **1Dパラメータ / その他** | トークン埋め込み層 ($W_{\text{emb}}$)、RMSNorm スケール係数 ($\gamma$)、バイアス成分、出力層ウェイト | **AdamW** |

#### Muonオプティマイザのメカニズム
2D行列に対して適用される **Muon** は、更新のステップサイズを直交化（Newton-Schulz反復による直交射影）することでパラメータを直交行列に近く保ち、効率的なパラメータ空間の探索を可能にする。これにより、特異値の崩壊を防ぎ、学習の収束速度が従来のAdamW単体と比べて格段に向上する。

#### AdamWオプティマイザのメカニズム
1Dパラメータ（スケール、バイアス、埋め込み）に対しては、Muonによる更新の恩恵を受けないか、あるいは数値的性質が直交化に適さないため、従来のL2デカプリングを伴う **AdamW** を使用する。
*   $\beta_1 = 0.9$, $\beta_2 = 0.95$
*   重み減衰 (Weight Decay): 0.1

### 3.3 計算効率化技術
*   **混合精度学習 (Mixed Precision)**: `bfloat16` (BF16) をネイティブで採用。これにより浮動小数点のオーバーフローやアンダーフローを防ぎつつ、VRAM使用量を半減させ計算速度を最大化する。
*   **勾配チェックポインティング (Gradient Checkpointing)**: メモリ節約のため、逆伝播時にトランスフォーマー層の中間アクティベーションを再計算する。これによりバッチサイズを拡大可能にしている。
