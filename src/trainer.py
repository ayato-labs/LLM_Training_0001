import torch
import torch.optim as optim
from .registry import MODEL_REGISTRY
from .modern_gpt import ModernGPT, ModernGPTConfig  # 明示的にインポート
from .normuon import NorMuon
from .wsd import WSDScheduler
from datasets import load_from_disk
import sys

def train_model(config_dict, trial=None):
    # Setup
    model_name = config_dict.get('model_name', 'modern_gpt')
    model_cls = MODEL_REGISTRY.get(model_name)
    
    # Instantiate model using its Config class
    model_config = ModernGPTConfig(**config_dict.get('model_config', {}))
    model = model_cls(model_config).cuda()
    
    # Optimizer selection
    optim_type = config_dict.get('optimizer', 'adamw')
    if optim_type == 'normuon':
        params_2d = [p for p in model.parameters() if p.ndim == 2]
        params_1d = [p for p in model.parameters() if p.ndim < 2]
        optimizer = NorMuon(params_2d, lr=config_dict.get('lr', 0.02))
        optimizer_1d = optim.AdamW(params_1d, lr=config_dict.get('lr', 0.002))
        def step():
            optimizer.step()
            optimizer_1d.step()
        def zero_grad():
            optimizer.zero_grad()
            optimizer_1d.zero_grad()
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config_dict.get('lr', 0.001))
        step = optimizer.step
        zero_grad = optimizer.zero_grad
    
    # Simple training loop
    model.train()
    total_loss = 0
    for i in range(5): 
        inputs = torch.randint(0, 1000, (4, 128)).cuda()
        targets = inputs.clone()
        
        zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        step()
        total_loss += loss.item()
    
    # Clean up
    del model, optimizer
    torch.cuda.empty_cache()
    
    return total_loss / 5
