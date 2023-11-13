import torch

DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SZ = 32
EPOCHS   = 15
LR       = 1e-4

def train():
    pass

def test():
    pass

if __name__=="__main__":
    print("Hello world!")
