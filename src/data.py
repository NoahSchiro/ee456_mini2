from torchvision.datasets import CIFAR10
from torchvision import transforms as T

def get_data():

    transform = T.Compose([
        T.ToTensor()
        # TODO: May want to also normalize the data
    ])

    train = CIFAR10("./data", train=True, transform=transform) 
    test  = CIFAR10("./data", train=False, transform=transform) 

    return train, test
