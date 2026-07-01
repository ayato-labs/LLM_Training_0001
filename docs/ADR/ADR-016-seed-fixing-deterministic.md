# ADR-016: 乱数シード完全固定（決定論的実行）

## ステータス
Accepted

## コンテキスト

### 現状の課題
論文レベルの再現性を確保するため、同じ設定・同じデータで実行すれば **必ず同じ結果** が得られることが必須である。

- 従来の `TrainingArguments` には `seed` パラメータが存在するが、明示的な設定は Trainer 委任
- `torch.use_deterministic_algorithms()` が未設定
- numpy/python random のシードが未固定
- cuDNN の benchmark モードが ON のまま（非決定論的挙動の原因）

---

## 意思決定

**`src/utils/set_seed.py` を新設し、学習開始時に全乱数源を固定する。**

### 固定対象
| 乱数源 | 固定方法 |
|--------|---------|
| Python `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch CPU | `torch.manual_seed(seed)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(seed)` |
| Python ハッシュ | `os.environ["PYTHONHASHSEED"] = str(seed)` |
| cuDNN | `torch.backends.cudnn.benchmark = False` |
| 決定論的アルゴリズム | `torch.use_deterministic_algorithms(True)` |
| CUBLAS | `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` |

### 使用方法
```python
from src.utils.set_seed import set_seed
set_seed(42, deterministic=True)  # 学習開始前に1回呼び出し
```

### 関連変更
- `train_model.py`: `set_seed()` をMLflow初期化前に呼び出し
- `TrainingArguments`: `seed=seed, deterministic=True` を明示

---

## 結果と影響

### 1. 再現性保証
- 同一設定・同一データ・同一環境で **完全に同じ損失曲線** が得られる
- 論文の「Reproducibility」要件を満たす

### 2. パフォーマンスへの影響
- `deterministic=True` + `cudnn.benchmark=False` により **約10-20%の速度低下** が発散する可能性
- VRAM 4GB環境では速度低下が許容範囲内であると判断（9秒/step → 10-11秒/step）

### 3. 注意事項
- `deterministic=True` は一部のCUDA演算でエラーを発生させることがある（未対応演算）
- その場合は `deterministic=False` にフォールバックし、`set_seed()` のみを適用
