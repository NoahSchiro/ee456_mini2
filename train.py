import torch
from torch.utils.data import DataLoader

from src.model import Model
from src.data import get_data 

# TODO: Device will be implemented on everything once we get to the optimization stage
DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SZ = 64
EPOCHS   = 15
LR       = 1e-3

def train(model, train_dl, optim, loss_fn):

    model.train()

    avg_loss = 0

    for batch, (imgs, labels) in enumerate(train_dl):

        optim.zero_grad()

        predictions = model(imgs)
        
        avg_loss += loss_fn(predictions, labels)

        if batch % 100 == 0:
            avg_loss /= 100
            print(f"Batch: {batch:4d}/{len(train_dl):4d} | Avg loss: {avg_loss:.5f}")
            avg_loss = 0

        optim.step()


@torch.no_grad()
def test(model, test_dl):

    model.eval()

    avg_loss = 0 
    correct = 0

    print(f"Testing...")
    
    for (imgs, labels) in test_dl:

        predictions = model(imgs)
        
        avg_loss += loss_fn(predictions, labels)

        _, predictions = torch.max(predictions, 1)
        correct += (predictions == labels).sum().item()

    print(f"End of epoch test complete")
    print(f"Accuracy: {correct / len(test_dl.dataset) * 100:3.2f}%")
    print(f"Avg_loss: {avg_loss / len(test_dl.dataset):3.2f}")



if __name__=="__main__":

    model   = Model()
    optim   = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters...")

    train_ds, test_ds = get_data()

    print(f"Loaded data...")
    print(f"Train size = {len(train_ds)}")
    print(f"Test size = {len(test_ds)}")

    # TODO: In optimization we need to pin memory and increase worker count
    train_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True)
    test_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True)

    for epoch in range(1, EPOCHS+1):

        print(f"Epoch {epoch}")
        train(model, train_dl, optim, loss_fn)
        test(model, test_dl)
