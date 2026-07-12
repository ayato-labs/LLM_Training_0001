def estimate_config_from_params(target_params: int) -> dict:
    """プロキシモデルの設定を計算（4GB GPU用に極小化）"""
    # プロキシモデルは本番の5%だが、最小3Mパラメータ
    # 語彙サイズを削減（32768 -> 8192）してメモリ節約
    vocab_size = 8192
    
    # L=4, H=256, n_head=8 で約 4.5M パラメータ
    # 埋め込み: 8192 * 256 = 2.1M パラメータ（32768->8192で削減）
    
    return {
        "n_layer": 4,
        "n_embd": 256,
        "n_head": 8,
        "n_kv_head": 2,
        "vocab_size": 8192,  # 32768 -> 8192 に削減
    }