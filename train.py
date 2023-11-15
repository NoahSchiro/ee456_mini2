import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR

from src.model import Model
from src.data import get_data 

# TODO: Device will be implemented on everything once we get to the optimization stage
DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CPUCORES = 8
BATCH_SZ = 64
EPOCHS   = 50
LR       = .1
LR_GAMMA = 0.975

scalar  = GradScaler()

def train(model, train_dl, optim, loss_fn):

    model.train()

    avg_loss = 0

    for batch, (imgs, labels) in enumerate(train_dl):

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optim.zero_grad()

        with autocast():
            predictions = model(imgs)
            loss = loss_fn(predictions, labels)

        scalar.scale(loss).backward()
        scalar.step(optim)
        scalar.update()
        
        avg_loss += loss

        if batch % 100 == 0 and batch != 0:
            avg_loss /= 100
            print(f"Batch: {batch:3d}/{len(train_dl):3d} | Avg loss: {avg_loss:.5f}")
            avg_loss = 0


@torch.no_grad()
def test(model, test_dl):

    model.eval()

    avg_loss = 0 
    correct = 0

    print(f"Testing...")
    
    for (imgs, labels) in test_dl:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        predictions = model(imgs)
        
        avg_loss += loss_fn(predictions, labels)

        _, predictions = torch.max(predictions, 1)
        correct += (predictions == labels).sum().item()

    print(f"End of epoch test complete")
    print(f"Accuracy: {correct / len(test_dl.dataset) * 100:3.2f}%")
    print(f"Avg_loss: {avg_loss / len(test_dl.dataset):3.5f}")



if __name__=="__main__":

    model   = Model().to(DEVICE)
    optim   = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = ExponentialLR(optim, gamma=LR_GAMMA)
    
    for param_group in optim.param_groups:
        print(param_group['lr'])

    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters...")

    train_ds, test_ds = get_data()

    print(f"Loaded data...")
    print(f"Train size = {len(train_ds)}")
    print(f"Test size = {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True, pin_memory=True, num_workers=CPUCORES)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SZ, shuffle=True, pin_memory=True, num_workers=CPUCORES)

    for epoch in range(1, EPOCHS+1):

        print(f"Epoch {epoch}")
        train(model, train_dl, optim, loss_fn)
        test(model, test_dl)
        scheduler.step()
        for param_group in optim.param_groups:
            print(param_group['lr'])
