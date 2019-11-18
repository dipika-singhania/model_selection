import argparse
import os
import torchvision
import torch
import numpy as np 
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from get_labels_to_vec import get_word_list_embeddings,add_vector_to_file
import pickle

def load_mnist(data_dir, transform=None, validation_split = 0.2, batch_size=256, add_feature=False):
    root = os.path.join(data_dir, 'MNIST')
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                          normalize, 
                                        ])
    download = True
    if os.path.exists(os.path.join(root,'processed/test.pt')) and os.path.exists(os.path.join(root,'processed/training.pt')):
        download = False
    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    
    if add_feature == True:
        image, label = train_dataset.__getitem__(0)
        _, img_h, img_w = image.shape
        train_size = len(train_dataset)
        num_classes = 10
        labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        word_embed = get_word_list_embeddings(labels)
        add_vector_to_file("Mnist",img_h, img_w, train_size,num_classes, word_embed)
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=False,
    )

    print("Got total number test data ", len(test_dataset))
    return (train_loader, valid_loader, test_loader, 10)


def load_FashionMNIST(data_dir, transform=None, validation_split = 0.2, batch_size=256, add_feature=False):
    root = os.path.join(data_dir, 'FashionMNIST')
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                          normalize, 
                                        ])
    download = True
    if os.path.exists(os.path.join(root,'processed/test.pt')) and os.path.exists(os.path.join(root,'processed/training.pt')):
        download = False
    # load the dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    valid_dataset = datasets.FashionMNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    if add_feature == True:
        image, label = train_dataset.__getitem__(0)
        _, img_h, img_w = image.shape
        train_size = len(train_dataset)
        num_classes = 10
        labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag"]
        word_embed = get_word_list_embeddings(labels)
        add_vector_to_file("FashionMnist",img_h, img_w, train_size,num_classes, word_embed)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=False,
    )

    print("Got total number test data ", len(test_dataset))
    return (train_loader, valid_loader, test_loader, 10)


def load_KMNIST(data_dir, transform=None, validation_split = 0.2, batch_size=256, add_feature=False):
    root = os.path.join(data_dir, 'KMNIST')
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                          normalize, 
                                        ])
    download = True
    if os.path.exists(os.path.join(root,'processed/test.pt')) and os.path.exists(os.path.join(root,'processed/training.pt')):
        download = False
    # load the dataset
    train_dataset = datasets.KMNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    valid_dataset = datasets.KMNIST(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    test_dataset = datasets.KMNIST(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]


    if add_feature == True:
        image, label = train_dataset.__getitem__(0)
        _, img_h, img_w = image.shape
        train_size = len(train_dataset)
        num_classes = 10
        labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        word_embed = get_word_list_embeddings(labels)
        add_vector_to_file("KMnist",img_h, img_w, train_size,num_classes, word_embed)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=False,
    )

    print("Got total number test data ", len(test_dataset))
    return (train_loader, valid_loader, test_loader, 10)


def load_cifar10(data_dir, transform=None, validation_split = 0.2, batch_size=256, add_feature=False):
    root = os.path.join(data_dir, 'cifar10')
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize, 
                                        ])
    download = True
    if os.path.exists(os.path.join(data_dir,'data/cifar-10-python.tar.gz')):
        download = False
    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if add_feature == True:
        image, label = train_dataset.__getitem__(0)
        _, img_h, img_w = image.shape
        train_size = len(train_dataset)
        num_classes = 10
        fp = open('data/cifar-10-batches-py/batches.meta','rb')
        labels = list(pickle.load(fp)['label_names'])
        word_embed = get_word_list_embeddings(labels)
        add_vector_to_file("Cifar-10",img_h, img_w, train_size,num_classes, word_embed)

    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=False,
    )

    print("Got total number test data ", len(test_dataset))
    return (train_loader, valid_loader, test_loader, 10)

def load_cifar100(data_dir, transform=None, validation_split = 0.2, batch_size=256, add_feature=False):
    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          normalize, 
                                        ])
    download = True
    if os.path.exists(os.path.join(data_dir,'data/cifar-100-python.tar.gz')):
        download = False
    # load the dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=download, transform=transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=download, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]


    if add_feature == True:
        image, label = train_dataset.__getitem__(0)
        _, img_h, img_w = image.shape
        train_size = len(train_dataset)
        num_classes = 100
        fp = open('data/cifar-100-python/meta','rb')
        labels = list(pickle.load(fp)['fine_label_names'])
        word_embed = get_word_list_embeddings(labels)
        add_vector_to_file("Cifar100",img_h, img_w, train_size,num_classes, word_embed)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=4, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=4, pin_memory=False,
    )

    print("Got total number test data ", len(test_dataset))
    return (train_loader, valid_loader, test_loader, 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    args = parser.parse_args()
    # load_mnist(args.data_dir, None, args.validation_split, args.batch_size, True)
    # load_KMNIST(args.data_dir, None, args.validation_split, args.batch_size, True)
    # load_FashionMNIST(args.data_dir, None, args.validation_split, args.batch_size, True)
    load_cifar10(args.data_dir, None, args.validation_split, args.batch_size, True)
    load_cifar100(args.data_dir, None, args.validation_split, args.batch_size, True)
    

