import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_

# Our code
from src.model import Model
from src.data import get_data 

# Data reporting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns

DEVICE   = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CPUCORES = 8
BATCH_SZ = 64
EPOCHS   = 50
LR       = .1
LR_GAMMA = 0.95
# MAX_CLIP = 10

scalar  = GradScaler()

# Some globally needed variables for reporting metrics
training_loss_history   = []
validation_loss_history = []
accuracy_history        = []
con_max                 = None
precision               = None
recall                  = None

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
        # clip_grad_norm_(model.parameters(), MAX_CLIP)
        scalar.step(optim)
        scalar.update()
        
        avg_loss += loss

        if batch % 100 == 0 and batch != 0:
            avg_loss /= 100
            training_loss_history.append(avg_loss.item())
            print(f"Batch: {batch:3d}/{len(train_dl):3d} | Avg loss: {avg_loss:.5f}")
            avg_loss = 0


@torch.no_grad()
def test(model, test_dl):

    model.eval()

    avg_loss = 0 
    correct = 0

    print(f"Testing...")
    all_preds = []
    all_labels = []
    
    for (imgs, labels) in test_dl:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        predictions = model(imgs)
        
        avg_loss += loss_fn(predictions, labels).item()

        _, predictions = torch.max(predictions, 1)
        correct += (predictions == labels).sum().item()

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    correct /= len(test_dl.dataset)
    correct *= 100
    avg_loss /= len(test_dl)

    print(f"End of epoch test complete")
    print(f"Accuracy: {correct:3.2f}%")
    print(f"Avg_loss: {avg_loss:3.5f}")
    
    validation_loss_history.append(avg_loss)
    accuracy_history.append(correct)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    global con_max, precision, recall
    # Confusion matrix computation
    con_max = confusion_matrix(all_labels, all_preds)
    # Compute precision and recall for each class
    precision = precision_score(all_labels, all_preds, average=None)
    recall    = recall_score(all_labels, all_preds, average=None)
    

def generate_graphs():
    total_batches = len(training_loss_history)
    epochs = np.linspace(1, EPOCHS, num=total_batches)

    # Training / validation graph
    plt.plot(epochs, training_loss_history, 'b', label='Training Loss')
    plt.plot(np.arange(1, EPOCHS+1), validation_loss_history, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig("./report/loss.png")
    plt.clf()

    # Accuracy graph
    plt.plot(np.arange(1, EPOCHS+1), accuracy_history, 'r', label='Accuracy')
    plt.title('Changes in accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.savefig("./report/accuracy.png")
    plt.clf()

    # Confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(con_max, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("./report/confusion_matrix.png")
    plt.clf()

    # Print precision and recall for each class
    global precision, recall
    for class_label, precision, recall in zip(range(10), precision, recall):
        print(f'Class {class_label}: Precision={precision:.4f}, Recall={recall:.4f}')

def display_images(model, test_ds):
    random_images = [test_ds[i][0] for i in range(10)]
    as_batch = torch.stack(random_images).to(DEVICE)

    probs, predictions = torch.max(model(as_batch).cpu(), 1)

    plt.figure(figsize=(15,8))
    for i in range(len(random_images)):
        plt.subplot(2, 5, i+1)
        plt.imshow(random_images[i].permute(1,2,0))
        plt.title(f"Prediction: {predictions[i]}; {probs[i]*100:.2f}%")
        plt.axis("off")

    plt.savefig("./report/example_predictions.png")
    plt.clf()


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
            print(f"{param_group['lr']:.5f}")

    # Report metrics
    generate_graphs()
    display_images(model, test_ds)

    # Save the model
    torch.save(model.state_dict(), "saved_CNN_small.pt")
