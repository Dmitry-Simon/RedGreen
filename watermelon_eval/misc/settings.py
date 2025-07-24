import os
from pathlib import Path
import torch

BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3

# Device configuration for Mac with MPS support
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

FIXED_WIDTH = 512
VERBOSE = False
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT   = PROJECT_ROOT / "watermelon_dataset"
