import torch

from src.model import Model

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

    # batch, channels, height, width
    random_img = torch.randn(32, 3, 16, 16)

    out = model(random_img)

    print(out[0])
