import torch
import torch.optim as optim

from src.logger import logger
from src.modern_gpt import ModernGPTConfig
from src.normuon import NorMuon
from src.registry import MODEL_REGISTRY


def train_model(config_dict, trial=None):
    try:
        model_name = config_dict.get("model_name", "modern_gpt")
        model_cls = MODEL_REGISTRY.get(model_name)

        model_config = ModernGPTConfig(**config_dict.get("model_config", {}))
        model = model_cls(model_config).cuda()

        optim_type = config_dict.get("optimizer", "adamw")
        if optim_type == "normuon":
            params_2d = [p for p in model.parameters() if p.ndim == 2]
            params_1d = [p for p in model.parameters() if p.ndim < 2]
            opt_2d = NorMuon(params_2d, lr=config_dict.get("lr", 0.02))
            opt_1d = optim.AdamW(params_1d, lr=config_dict.get("lr", 0.002))

            def step():
                opt_2d.step()
                opt_1d.step()

            def zero_grad():
                opt_2d.zero_grad()
                opt_1d.zero_grad()

            optimizer = opt_2d
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config_dict.get("lr", 0.001))
            step = optimizer.step
            zero_grad = optimizer.zero_grad

        model.train()
        total_loss = 0
        for _i in range(5):
            inputs = torch.randint(0, 1000, (4, 128)).cuda()
            targets = inputs.clone()

            zero_grad()
            logits, loss = model(inputs, targets=targets)
            loss.backward()
            step()
            total_loss += loss.item()

        avg_loss = total_loss / 5
        logger.debug(f"HPO proxy train: avg_loss={avg_loss:.6f}")

        del model, optimizer
        torch.cuda.empty_cache()

        return avg_loss

    except Exception as e:
        logger.error(f"HPO proxy training failed: {e}", exc_info=True)
        raise
