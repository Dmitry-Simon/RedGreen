import os
from pathlib import Path
import torch

BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIDTH = 512
VERBOSE = False
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT   = PROJECT_ROOT / "watermelon_dataset"

