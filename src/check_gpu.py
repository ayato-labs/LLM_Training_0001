import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
else:
    print("CUDA is NOT available. Checking torch/cuda installation...")
    # Add extra debug info
    print(f"Is CUDA compiled: {torch.version.cuda is not None}")
