from dataset_loaders.image_classification.cars import load_cars
from dataset_loaders.image_classification.traffic_signal import load_traffic
from dataset_loaders.image_classification.flower import load_flower
from dataset_loaders.image_classification.pytorch_datasets import load_mnist, load_FashionMNIST, load_KMNIST, load_cifar10, load_cifar100

DATA_DICTIONARY = {'cars': load_cars,
                   'flower': load_flower,
                   'mnist': load_mnist,
                   'kmnist': load_KMNIST,
                   'fashionmnist': load_FashionMNIST,
                   'traffic': load_traffic,
                   'cifar10': load_cifar10,
                   'cifar100': load_cifar100}


def return_dataloader(data_name, root, batch_size):
    '''
    :param data_name: Name of the dataset
    :return: a dictionary with keys 'name', 'train', 'valid', 'test'
    '''
    data_dict = DATA_DICTIONARY
    if data_name not in data_dict:
        print('Data name not present in dataset dictionary! Available datasets are')
        for i, name in enumerate(data_dict.keys(), 1):
            print('%d. %s'%(i, name))
        print('''
        If you intend to add a new dataset, please add a dataloader function in \'dataset_loaders\\image_classification\\\'
        and add dictionary entry in \'dataset_loaders\\image_classification\\data_config.py\'
        ''')
    else:
        load_fn = data_dict[data_name]
        return load_fn(data_dir=root, batch_size=batch_size)
