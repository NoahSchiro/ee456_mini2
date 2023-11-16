from torch import nn
from torch.nn import functional as F

# Define the model (loosely based on AlexNet)
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=True)
        self.p1 = nn.MaxPool2d(3)

        self.c2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        self.p2 = nn.MaxPool2d(3)

        self.c3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.p3 = nn.MaxPool2d(3)

        self.drop = nn.Dropout(p=0.1)

        self.flatten = nn.Flatten()

        # MLP at the end to classify 
        self.l1 = nn.Linear(64, 1024)
        self.l2 = nn.Linear(1024, 10)
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img):

        img = self.c1(img)
        img = F.relu(img, inplace=True)
        img = self.p1(img)

        img = self.c2(img)
        img = F.relu(img, inplace=True)
        img = self.p2(img)

        img = self.c3(img)
        img = F.relu(img, inplace=True)
        img = self.p3(img)

        img = self.drop(img)
        
        vec = self.flatten(img)
        # Now it is 253952 element long vector

        # Classify using MLP
        vec = self.l1(vec)
        vec = F.relu(vec, inplace=True)
        vec = self.l2(vec)
        
        logits = self.softmax(vec)
        
        return logits 

