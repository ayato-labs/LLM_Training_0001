"""
ADR-018/019/021 実装テスト (現行 Hydra defaults 構成対応版)
- BF16 対応
- 150M/1024ctx アーキテクチャ
- Vocab 64k + end_of_story
- base_config / scaling_config / hpo_config の合成 (Hydra defaults)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


def _deep_merge(base: dict, override: dict) -> dict:
    """Hydra defaults と同じネスト辞書マージ (後勝ち)"""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_merged_raw() -> dict:
    """configs/ の 3 ファイルを Hydra defaults 順 (base < scaling < hpo) で合成"""
    import yaml

    merged: dict = {}
    for name in ("base_config.yaml", "scaling_config.yaml", "hpo_config.yaml"):
        path = PROJECT_ROOT / "configs" / name
        assert path.exists(), f"{path} not found"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        assert isinstance(cfg, dict), f"{name} must be a dict"
        merged = _deep_merge(merged, cfg)
    return merged


def _load_normalized_config() -> dict:
    """load_config 相当 (OmegaConf 経由) の正規化設定を返す"""
    from omegaconf import OmegaConf

    from src.training.config import load_config

    return load_config(OmegaConf.create(_load_merged_raw()))


# ============================================================
# 1. 合成設定の値テスト
# ============================================================
class TestTrainingConfig:
    def test_seed(self):
        """ADR: seed は base_config の 42"""
        assert _load_merged_raw()["seed"] == 42

    def test_precision_default(self):
        """ADR-018: デフォルト precision は bf16"""
        assert _load_merged_raw()["precision"] == "bf16"

    def test_target_params_from_scaling(self):
        """ADR-019: ターゲットパラメータ数は scaling_config の target_params"""
        normalized = _load_normalized_config()
        expected = _load_merged_raw()["model"]["target_params"]
        assert normalized["model_params"]["n_params"] == expected

    def test_seq_len(self):
        """ADR-019: シーケンス長は base_config の 512"""
        normalized = _load_normalized_config()
        assert normalized["hpo"]["seq_len"] == 512
        assert normalized["seq_len"] == 512

    def test_vram_limit_default(self):
        """ADR-019: VRAM制限は自動検出される"""
        from src.common.vram_estimator import detect_vram

        vram = detect_vram()
        assert vram > 0, f"VRAM should be detected, got {vram}"

    def test_hpo_values_merged(self):
        """HPO 成果物 (max_lr_2d 等) が training セクションへ反映される"""
        normalized = _load_normalized_config()
        raw = _load_merged_raw()
        assert normalized["max_lr_2d"] == raw["training"]["max_lr_2d"]
        assert normalized["weight_decay"] == raw["training"]["weight_decay"]
        assert normalized["warmup_ratio"] == raw["training"]["warmup_ratio"]

    def test_max_steps_from_computable_tokens(self):
        """Chinchilla metadata.computable_tokens から max_steps が動的計算される"""
        normalized = _load_normalized_config()
        raw = _load_merged_raw()
        expected = int(
            raw["metadata"]["computable_tokens"]
            // (raw["training"]["batch_size_seqs"] * raw["training"]["seq_len"])
        )
        assert normalized["max_steps"] == expected
        assert normalized["max_steps"] > 10_000


# ============================================================
# 2. configs/config.yaml (Hydra defaults) テスト
# ============================================================
class TestConfigYAML:
    def test_defaults_order(self):
        """config.yaml の defaults が base/scaling/hpo の順"""
        import yaml

        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        defaults = self.config.get("defaults", [])
        assert defaults == ["base_config", "scaling_config", "hpo_config", "_self_"]

    def test_base_config_architecture_defaults(self):
        """base_config.yaml にアーキテクチャ既定値が存在する"""
        import yaml

        path = PROJECT_ROOT / "configs" / "base_config.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["seed"] == 42
        assert cfg["precision"] == "bf16"
        assert cfg["training"]["seq_len"] == 512
        assert cfg["training"]["packing"] is True

    def test_scaling_config_architecture(self):
        """scaling_config.yaml にアーキテクチャが定義される"""
        import yaml

        path = PROJECT_ROOT / "configs" / "scaling_config.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        llama = cfg["model"]["llama"]
        assert cfg["model"]["target_params"] > 0
        assert llama["hidden_size"] > 0
        assert llama["num_hidden_layers"] > 0
        assert llama["num_attention_heads"] % llama["num_key_value_heads"] == 0
        assert llama["vocab_size"] >= 32000

    def test_model_params_count(self):
        """モデルパラメータ数の概算が target_params と同程度 (±50%)"""
        llama = _load_merged_raw()["model"]["llama"]
        target = _load_merged_raw()["model"]["target_params"]
        hidden = llama["hidden_size"]
        layers = llama["num_hidden_layers"]
        vocab = llama["vocab_size"]
        intermediate = llama["intermediate_size"]
        # LLaMA 概算: attention(4H^2/L) + FFN(3HI) + embedding (tied)
        est_params = layers * (4 * hidden * hidden + 3 * hidden * intermediate) + vocab * hidden
        assert 0.5 * target < est_params < 1.5 * target, (
            f"Estimated params {est_params / 1e6:.1f}M vs target {target / 1e6:.1f}M"
        )


# ============================================================
# 3. 依存関係テスト
# ============================================================
class TestDependencies:
    @pytest.fixture(autouse=True)
    def setup(self):
        import tomllib

        config_path = PROJECT_ROOT / "pyproject.toml"
        with open(config_path, "rb") as f:
            self.pyproject = tomllib.load(f)

    def test_torch_version(self):
        deps = self.pyproject["project"]["dependencies"]
        torch_deps = [d for d in deps if "torch" in d.lower() and "torchvision" not in d.lower()]
        assert len(torch_deps) > 0
        assert ">=2.1.0" in torch_deps[0]


# ============================================================
# 4. Python バージョンテスト
# ============================================================
class TestPythonVersion:
    def test_python_version_file(self):
        version_file = PROJECT_ROOT / ".python-version"
        assert version_file.exists()
        content = version_file.read_text().strip()
        assert content == "3.12"


# ============================================================
# 5. ADR ファイル存在テスト
# ============================================================
class TestADRFiles:
    def test_adr018_exists(self):
        assert (PROJECT_ROOT / "docs" / "ADR" / "001-bf16-adoption.md").exists()

    def test_adr019_exists(self):
        assert (PROJECT_ROOT / "docs" / "ADR" / "003-150m-1024ctx-architecture.md").exists()

    def test_adr020_exists(self):
        assert (PROJECT_ROOT / "docs" / "ADR" / "004-packed-sequence.md").exists()

    def test_adr021_exists(self):
        assert (PROJECT_ROOT / "docs" / "ADR" / "005-vocab-64k-end-of-story.md").exists()


# ============================================================
# 6. アーキテクチャ整合性テスト
# ============================================================
class TestArchitectureConsistency:
    def test_params_consistency(self):
        """scaling_config の target_params が正規化後 model_params に反映される"""
        normalized = _load_normalized_config()
        expected = _load_merged_raw()["model"]["target_params"]
        assert normalized["model_params"]["n_params"] == expected

    def test_gqa_consistency(self):
        """GQA 制約: num_attention_heads は num_key_value_heads の倍数"""
        normalized = _load_normalized_config()
        heads = normalized["model_params"]["num_attention_heads"]
        kv_heads = normalized["model_params"]["num_key_value_heads"]
        assert kv_heads >= 1
        assert heads % kv_heads == 0

    def test_precision_consistency(self):
        """precision が bf16"""
        normalized = _load_normalized_config()
        assert normalized["precision"] == "bf16"


# ============================================================
# 7. モデル初期化テスト (torch必要)
# ============================================================
class TestModelInit:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import torch

            self.torch = torch
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            pytest.skip("torch not installed")

    def test_llama_config_creation(self):
        """LlamaConfig が正しく作成できる"""
        from transformers import LlamaConfig

        config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=16,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=64000,
            max_position_embeddings=2048,
        )
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 16
        assert config.num_attention_heads == 8
        assert config.vocab_size == 64000
        assert config.max_position_embeddings == 2048

    def test_model_instantiation_cpu(self):
        """LlamaForCausalLM が CPU で初期化できる"""
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=16,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=64000,
            max_position_embeddings=2048,
        )
        model = LlamaForCausalLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        assert 60_000_000 < param_count < 140_000_000, (
            f"Model params should be ~100M, got {param_count / 1e6:.1f}M"
        )

    def test_model_forward_cpu(self):
        """CPU で forward pass が通る"""
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=64000,
            max_position_embeddings=2048,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        input_ids = torch.randint(0, 64000, (1, 128))
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits.shape == (1, 128, 64000)

    def test_model_bf16_forward_gpu(self):
        """GPU で bf16 forward pass が通る"""
        if not self.cuda_available:
            pytest.skip("CUDA not available")
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=64000,
            max_position_embeddings=2048,
        )
        model = LlamaForCausalLM(config).to(torch.bfloat16).cuda()
        model.eval()
        input_ids = torch.randint(0, 64000, (1, 128)).cuda()
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits.dtype == torch.bfloat16
        assert outputs.logits.shape == (1, 128, 64000)

    def test_model_resize_token_embeddings(self):
        """特殊トークン追加後の resize が動作"""
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=64000,
            max_position_embeddings=2048,
        )
        model = LlamaForCausalLM(config)
        original_vocab = model.config.vocab_size
        model.resize_token_embeddings(original_vocab + 9)
        assert model.config.vocab_size == original_vocab + 9
        assert model.get_input_embeddings().weight.shape[0] == original_vocab + 9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
