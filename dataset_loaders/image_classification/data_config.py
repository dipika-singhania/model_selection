from .cars import load_cars
from .traffic_signal import load_traffic
from .flower import load_flower
from .pytorch_datasets import load_mnist, load_FashionMNIST, load_KMNIST, load_cifar10, load_cifar100

DATA_DICTIONARY = {'cars': load_cars,
                   'flowers': load_flower,
                   'mnist': load_mnist,
                   'kmnist': load_KMNIST,
                   'fashion-mnist': load_FashionMNIST,
                   'traffic': load_traffic,
                   'cifar10': load_cifar10,
                   'cifar100': load_cifar100}
