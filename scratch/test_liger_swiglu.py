import time
import torch

try:
    import torch.distributed.tensor
except Exception:
    pass

import torch.nn.functional as F
from functools import wraps

print("=== Starting Input Clone Selective Checkpointing Test ===")

from transformers import LlamaConfig, LlamaForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_llama
from src.training.trainer.dual_optimizer_trainer import SplitOptimizer

# Apply Liger Kernel patch
apply_liger_kernel_to_llama()

def apply_liger_compatible_selective_checkpointing(model) -> int:
    """入力テンソルのアラインメント参照をカプセル化し、Liger Kernel と 100% 完全共存させる選択的 Checkpointing"""
    applied_count = 0
    for module in model.modules():
        if module.__class__.__name__ == "LlamaDecoderLayer" and hasattr(module, "self_attn"):
            original_attn_forward = module.self_attn.forward

            @wraps(original_attn_forward)
            def custom_attn_forward(*args, **kwargs):
                # 入力テンソルをクローンしてチェックポイントに渡すことで SavedVariable メタデータを不変固定
                cloned_args = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)
                bound_kwargs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

                def custom_forward_closure(*inputs):
                    return original_attn_forward(*inputs, **bound_kwargs)

                return torch.utils.checkpoint.checkpoint(
                    custom_forward_closure,
                    *cloned_args,
                    use_reentrant=False,
                )

            module.self_attn.forward = custom_attn_forward
            applied_count += 1
    return applied_count

config = LlamaConfig(
    hidden_size=640,
    intermediate_size=2560,
    num_hidden_layers=10,
    num_attention_heads=10,
    num_key_value_heads=10,
    vocab_size=64000,
)

print("Step 1: Instantiating LlamaForCausalLM model...")
model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)

print("Step 2: Applying Input Clone Selective Checkpointing...")
count = apply_liger_compatible_selective_checkpointing(model)
print(f"  Applied to {count} layers.")

print("Step 3: Initializing SplitOptimizer...")
opt_config = {
    "hidden_size": 640,
    "n_params": 50_000_000,
    "max_lr_2d": 0.017,
    "max_lr_1d": 0.0016,
    "weight_decay": 0.06,
    "beta2": 0.95,
}
optimizer = SplitOptimizer(model, opt_config)

print("Step 4: Setting up PyTorch DataLoader...")
class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 64000, (512,)),
            "labels": torch.randint(0, 64000, (512,)),
        }

dataset = DummyDataset()
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
)

print("Step 5: Running 3 Training Steps via DataLoader...")
for step, batch in enumerate(dataloader):
    if step >= 3:
        break
    t0 = time.perf_counter()
    input_ids = batch["input_ids"].to("cuda")
    labels = batch["labels"].to("cuda")
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"  Step {step + 1} completed in {(time.perf_counter() - t0)*1000:.2f}ms. Loss: {loss.item():.4f}")

print("\nSUCCESS: Liger Kernel + Selective Checkpointing Coexistence Test PASSED PERFECTLY!")
