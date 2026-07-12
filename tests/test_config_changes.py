"""
ADR-018/019/021 実装テスト
- BF16 対応
- 150M/1024ctx アーキテクチャ
- Vocab 64k + end_of_story
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


# ============================================================
# 1. config.py の設定値テスト
# ============================================================
class TestTrainingConfig:
    def test_target_params(self):
        """ADR-019: ターゲットパラメータ数は 150M"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["model_params"]["n_params"] == 150_000_000

    def test_seq_len(self):
        """ADR-019: シーケンス長は 1024"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["hpo"]["seq_len"] == 1024

    def test_precision_default(self):
        """ADR-018: デフォルト precision は bf16"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["precision"] == "bf16"

    def test_vram_limit_default(self):
        """ADR-019: VRAM制限は自動検出される"""
        from src.config import detect_vram

        vram = detect_vram()
        assert vram > 0, f"VRAM should be detected, got {vram}"


# ============================================================
# 3. configs/config.yaml テスト
# ============================================================
class TestConfigYAML:
    @pytest.fixture(autouse=True)
    def setup(self):
        import yaml

        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_model_architecture(self):
        """ADR-019: 150M アーキテクチャ"""
        model = self.config["model"]
        assert model["target_params"] == 150_000_000
        assert model["llama"]["hidden_size"] == 768
        assert model["llama"]["num_hidden_layers"] == 12
        assert model["llama"]["num_attention_heads"] == 12
        assert model["llama"]["intermediate_size"] == 3072

    def test_rope_theta(self):
        """ADR-019: RoPE theta が 1024ctx に対応"""
        assert self.config["model"]["llama"]["rope_theta"] == 10000.0

    def test_seq_len(self):
        """ADR-019: シーケンス長 1024"""
        assert self.config["training"]["seq_len"] == 1024

    def test_vocab_size(self):
        """ADR-021: vocab 64k"""
        assert self.config["tokenizer"]["vocab_size"] == 64000

    def test_special_tokens(self):
        """ADR-021: end_of_story トークン存在"""
        tokens = self.config["tokenizer"]["special_tokens"]
        assert "<|end_of_story|>" in tokens
        assert "<|start_of_story|>" in tokens
        assert "<|start_of_metadata|>" in tokens
        assert "<|end_of_metadata|>" in tokens

    def test_hardware_precision(self):
        """ADR-018: precision = bf16"""
        assert self.config["hardware"]["precision"] == "bf16"

    def test_hardware_vram(self):
        """ADR-019: VRAM 4GB"""
        assert self.config["hardware"]["vram_limit_gb"] is not None

    def test_model_params_count(self):
        """100M モデルのパラメータ数概算"""
        model = self.config["model"]["llama"]
        # LLaMA: 12 * L * H^2 (embedding/HEAD除く)
        est_params = 12 * model["num_hidden_layers"] * (model["hidden_size"] ** 2)
        # + embedding (vocab * hidden)
        est_params += self.config["tokenizer"]["vocab_size"] * model["hidden_size"]
        # + FFN (2 * intermediate * hidden per layer)
        est_params += (
            2 * model["intermediate_size"] * model["hidden_size"] * model["num_hidden_layers"]
        )
        # 目安: 100M ± 50%
        assert 50_000_000 < est_params < 150_000_000, (
            f"Estimated params should be ~100M, got {est_params / 1e6:.1f}M"
        )


# ============================================================
# 5. pyproject.toml 依存関係テスト
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
# 6. Python バージョンテスト
# ============================================================
class TestPythonVersion:
    def test_python_version_file(self):
        version_file = PROJECT_ROOT / ".python-version"
        assert version_file.exists()
        content = version_file.read_text().strip()
        assert content == "3.12"


# ============================================================
# 7. ADR ファイル存在テスト
# ============================================================
class TestADRFiles:
    def test_adr018_exists(self):
        assert (PROJECT_ROOT / "docs" / "adr" / "ADR-018-bf16-adoption.md").exists()

    def test_adr019_exists(self):
        assert (PROJECT_ROOT / "docs" / "adr" / "ADR-019-150m-1024ctx-architecture.md").exists()

    def test_adr020_exists(self):
        assert (PROJECT_ROOT / "docs" / "adr" / "ADR-020-packed-sequence.md").exists()

    def test_adr021_exists(self):
        assert (PROJECT_ROOT / "docs" / "adr" / "ADR-021-vocab-64k-end-of-story.md").exists()


# ============================================================
# 8. アーキテクチャ整合性テスト
# ============================================================
class TestArchitectureConsistency:
    def test_params_consistency(self):
        """config.yaml の target_params が正しく読み込まれる"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["model_params"]["n_params"] == 150_000_000

    def test_seq_len_consistency(self):
        """config.yaml の seq_len が正しく読み込まれる"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["hpo"]["seq_len"] == 1024

    def test_precision_consistency(self):
        """config.yaml の precision が bf16"""
        from src.config import load_config, resolve_config_path

        config_path = resolve_config_path(None)
        config = load_config(config_path)
        assert config["precision"] == "bf16"


# ============================================================
# 9. モデル初期化テスト (torch必要)
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
        # 100M ± 40% (embedding/FFN含む)
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
        # 9特殊トークン追加
        model.resize_token_embeddings(original_vocab + 9)
        assert model.config.vocab_size == original_vocab + 9
        assert model.get_input_embeddings().weight.shape[0] == original_vocab + 9


# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
