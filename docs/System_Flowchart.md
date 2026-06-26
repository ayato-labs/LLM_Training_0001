# システム処理フロー図 (CPU/GPU役割分担)

本システムは、RTX 3050 (VRAM 4GB) 環境において「計算効率」と「小説知識密度」を最大化するため、CPUとGPUの処理を厳密に分離しています。

## 処理フロー図
```mermaid
graph TD
    subgraph CPU ["CPU (Host Memory / RAM / I/O)"]
        A[SQLite: 小説データ抽出] --> B[クレンジング/正規化]
        B --> C[BPE Tokenization]
        C --> D[バイナリキャッシュ保存]
        
        E[学習データローダー: I/O & Streaming] --> F[バッチ構築/Collator]
        F -->|オフロード解除| G[パラメータ/Optimizer退避]
        
        K[物語メモリ (RAG/外部ストレージ)] --> L[コンテキスト・プロンプト構築]
    end

    subgraph GPU ["GPU (VRAM / Tensor Cores)"]
        H[テンソル計算 (Forward Pass)]
        I[勾配計算 (Backward Pass)]
        J[FlashAttention / SwiGLU / RMSNorm]
    end

    %% Training Path
    F -->|Tensor転送| H
    H --> J
    J --> I
    I -->|勾配転送| G
    G -->|重み更新| H
    
    %% Inference Path
    L -->|Prompt Input| H
    H -->|Logits出力| CPU
    CPU -->|Decoding/Sampling| CPU

    style CPU fill:#f9f9f9,stroke:#333,stroke-width:2px
    style GPU fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
```

## 役割分担詳細

### 1. CPU (Host Memory / I/O)
* **データ前処理**: SQLiteからの抽出、ノイズ除去、Tokenize、ディスクへのArrow形式キャッシュ保存。学習のI/Oボトルネックを解消します。
* **メモリマネジメント (ZeRO-Offload)**: VRAMに収まりきらないモデルのパラメータ重みやオプティマイザの計算状態（Adamのモーメンタム等）を保持。
* **階層的推論 (RAG)**: 長編物語のコンテキストを維持するためのメモリ（プロット・キャラ設定）の管理と、推論時の動的プロンプト注入。

### 2. GPU (VRAM / Tensor Cores)
* **行列演算**: 畳み込みや線形変換などの行列演算をFP16で実行。
* **計算最適化**: FlashAttention-2を用いたAttention演算の高速化。
* **中間活性化 (Activation)**: 勾配チェックポインティングにより、計算に必要な最小限の中間データのみをVRAMに保持。
