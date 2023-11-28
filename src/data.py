from torchvision.datasets import CIFAR10
from torchvision import transforms as T

class_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def get_data():

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = CIFAR10("./data", train=True, transform=transform) 
    test  = CIFAR10("./data", train=False, transform=transform) 

    return train, test
