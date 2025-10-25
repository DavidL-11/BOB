import torch

"""
This script converts a full checkpoint (with optimizer, scheduler, etc.) to a state_dict-only checkpoint.
The EMA (Exponential Moving Average) weights are preferred if available.
"""

CHECKPOINT_PATH = "BOB_dfine_V3"

# Load your big checkpoint
ckpt = torch.load(f"runs/detect/{CHECKPOINT_PATH}/best_stg2.pth", map_location="cpu")

# Print the keys to understand the structure
print(ckpt.keys())

# Prefer EMA weights if available
if "ema" in ckpt and ckpt["ema"] is not None:
    state_dict = ckpt["ema"]["module"]
else:
    state_dict = ckpt["model"]

# Save only the model weights (no optimizer, no scheduler)
torch.save(state_dict, f"runs/detect/{CHECKPOINT_PATH}/{CHECKPOINT_PATH}.pt")