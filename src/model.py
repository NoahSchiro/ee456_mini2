from torch import nn
from torch.nn import functional as F

# Define the model (loosely based on AlexNet)
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(64)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 248, kernel_size=3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(248)

        self.drop = nn.Dropout2d(0.5)

        self.flatten = nn.Flatten()

        # MLP at the end to classify
        self.l1 = nn.Linear(248*16*16, 4096)
        self.l2 = nn.Linear(4096, 10)
    
    def forward(self, img):

        img = self.c1(img)
        img = self.b1(img)
        img = F.relu(img, inplace=True)

        img = self.c2(img)
        img = self.b2(img)
        img = F.relu(img, inplace=True)

        img = self.c3(img)
        img = self.b3(img)
        img = F.relu(img, inplace=True)
        
        img = self.drop(img)
        # By this point, our image is 248 x 16 x 16
        
        vec = self.flatten(img)
        # Now it is 63488 element long vector

        # Classify using MLP
        vec = self.l1(vec)
        vec = F.relu(vec, inplace=True)
        vec = self.l2(vec)

        logits = F.softmax(vec, dim=1)

        return logits

