import torch
from torch.utils.data import DataLoader

from src.model import Model
from src.data import get_data 

# Debugging
import matplotlib.pyplot as plt

DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SZ = 32
EPOCHS   = 15
LR       = 1e-4

def train():
    pass

def test():
    pass

if __name__=="__main__":

    model = Model()

    train_ds, test_ds = get_data()

    train_dl = DataLoader(train_ds, batch_size=BATCH_SZ)

    # Debugging
    for (img, label) in train_dl:
        out = model(img)

        print(out)

        exit(0)
