import inspect
import os.path

import torch
from torchvision import datasets, transforms


def load_dataset(dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    
    train_data, test_data = None, None
    
    if dataset.upper() == "MNIST":
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
    if dataset.upper() == "FMNIST":
        train_data = datasets.FashionMNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.FashionMNIST(root=datadir, train=False, transform=transform, download=True)
        
    if dataset.upper() == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_data, test_data


def split_data(train_data, clients):
    split_arr = [int(len(train_data) / len(clients)) for _ in range(len(clients))]
    rem_data = len(train_data) - (len(clients) * int(len(train_data) / len(clients)))
    if rem_data > 0:
        split_arr[-1] = split_arr[-1] + rem_data
    
    splitted_data = torch.utils.data.random_split(train_data, split_arr)
    clients_data = {client: splitted_data[index] for index, client in enumerate(clients)}

    return clients_data


def load_client_data(clients_data, batch_size, test_ratio=None, **kwargs):
    train_loaders = {}
    test_loaders = {}

    if test_ratio is None:
        for client, data in clients_data.items():
            train_loaders[client] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
            
    else:
        for client, data in clients_data.items():
            train_test = torch.utils.data.random_split(data, [int(len(data) * (1-test_ratio)), 
                                                              int(len(data) * test_ratio)])

            train_loaders[client] = torch.utils.data.DataLoader(train_test[0], batch_size=batch_size, shuffle=True, **kwargs)
            test_loaders[client] = torch.utils.data.DataLoader(train_test[1], batch_size=batch_size, shuffle=True, **kwargs)

    return train_loaders, test_loaders