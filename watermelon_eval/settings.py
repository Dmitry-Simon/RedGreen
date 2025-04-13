import torch

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_WIDTH = 300
VERBOSE = False