import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name}")
    print(f"vram_gb={props.total_memory / 1024**3:.1f}")
