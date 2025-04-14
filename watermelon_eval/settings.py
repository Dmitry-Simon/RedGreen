import os

import torch

BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIDTH = 300
VERBOSE = False

