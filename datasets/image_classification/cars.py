#
import tarfile
import os
import tempfile
import pickle
import numpy as np
import csv
import shutil
from tqdm import tqdm
from itertools import chain
from PIL import Image
import math

from datasets.image_classification.utils import download_dataset_from_url

import sys
import torch
# from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import argparse
import io
from torch.utils.data.dataloader import default_collate
from get_labels_to_vec import get_word_list_embeddings, add_vector_to_file
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)

class CarsDataset(Dataset):
    def __init__(self, cars_annotations, data_dir, car_names, transform=None, no_labels=False):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.car_annotations = cars_annotations

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

        self.no_labels = no_labels

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        if len(np.array(image).shape) < 2:
            return None
        if self.no_labels is not True:
            car_class = self.car_annotations[idx][-2][0][0] - 1
        else:
            car_class = -1

        if self.transform:
            try:
                image = self.transform(image)
            except:
                return None
        return image, car_class

    def get_num_of_class(self):
        return len(self.car_names)

    def get_cars_features(self):
        img_name = os.path.join(self.data_dir, self.car_annotations[0][-1][0])
        img_height, img_width, _ = np.array(Image.open(img_name)).shape
        total_train_size = len(self.car_annotations)
        num_of_classes = self.get_num_of_class()
        dataset_feature = get_word_list_embeddings(self.car_names) 
        add_vector_to_file('cars', img_height, img_width, total_train_size, num_of_classes, dataset_feature)

    def get_label_dict(self):
        labels_dict = {}
        for i, cars_name in enumerate(self.car_names):
            labels_dict[i] = cars_name[0]
        return labels_dict

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):
        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()


def download_cars_data(dataset_train_url, dataset_test_url, dataset_annotations_url, out_train_dataset_path,
                       out_test_dataset_path, out_meta_csv_path, data_dir):
    '''
        Loads and converts an image dataset of the cars dataset 
        Refer to https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

        :param str train_dataset_url: URL to download the Python version of the dataset
        :param str dataset_test_url: URL to download the Python version of the dataset
        :param str dataset_annotations_url: URL to download the Python version of the dataset
        :param str out_train_dataset_path: Path to save the output train dataset file
        :param str out_test_dataset_path: Path to save the output validation dataset file
        :param str out_meta_csv_path: Path to save the output dataset metadata .CSV file
    '''
    print('Downloading dataset archive...')
    if not os.path.exists(out_train_dataset_path):
        dataset_train_zip_file_path = download_dataset_from_url(dataset_train_url)
        tf = tarfile.open(dataset_train_zip_file_path)
        tf.extractall(data_dir)

    if not os.path.exists(out_test_dataset_path):
        dataset_test_zip_file_path = download_dataset_from_url(dataset_test_url)
        tf = tarfile.open(dataset_test_zip_file_path)
        tf.extractall(data_dir)

    if not os.path.exists(out_meta_csv_path):
        dataset_annotations_url_path = download_dataset_from_url(dataset_annotations_url)
        tf = tarfile.open(dataset_annotations_url_path)
        tf.extractall(data_dir)


def load_cars(data_dir, transform=None, validation_split = 0.2, batch_size=256):
    train_dataset_path = os.path.join(data_dir, 'cars_train')
    test_dataset_path = os.path.join(data_dir, 'cars_test')
    meta_csv_path = os.path.join(data_dir, 'devkit')
    train_dataset_url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
    test_dataset_url = 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz'
    meta_datset_url = 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
    train_batch_size = batch_size
    test_batch_size = batch_size
    train_workers = 4
    test_workers = 4
    download_cars_data(train_dataset_url, test_dataset_url, meta_datset_url, train_dataset_path, test_dataset_path,
                       meta_csv_path, data_dir)
    print('Loading datasets into memory...')
    
    full_data_set = scipy.io.loadmat(os.path.join(meta_csv_path, 'cars_train_annos.mat'))
    full_cars_anno = full_data_set['annotations'][0]
    total_len_cars = len(full_cars_anno)
    valid_split = int(validation_split * total_len_cars)
    train_cars_anno, valid_cars_anno = full_cars_anno[:-valid_split], full_cars_anno[-valid_split:]



    full_test_set = scipy.io.loadmat(os.path.join(meta_csv_path, 'cars_test_annos.mat'))
    test_cars_anno = full_data_set['annotations'][0]

    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                           transforms.Scale((224, 224)),
                                           transforms.ToTensor(),
                                           normalize, 
                                       ])
    cars_data_train = CarsDataset(train_cars_anno, train_dataset_path, os.path.join(meta_csv_path, 'cars_meta.mat'), transform, False)

    cars_data_val = CarsDataset(valid_cars_anno, train_dataset_path, os.path.join(meta_csv_path, 'cars_meta.mat'), transform, False)

    cars_data_test = CarsDataset(test_cars_anno, test_dataset_path, os.path.join(meta_csv_path, 'cars_meta.mat'), transform, True)

    get_ele = cars_data_train.__getitem__(1)
    print("Got element with label = ", get_ele[1])
    # for i in range(len(cars_data_test)):
    #     get_ele = cars_data_test.__getitem__(i)
    #     print(get_ele)
    features_list = cars_data_train.get_cars_features()


    trainloader = DataLoader(cars_data_train, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers, collate_fn=my_collate)
    print("Train data set length:", len(cars_data_train))

    validloader = DataLoader(cars_data_val, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers, collate_fn=my_collate)
    print("Validation data set length:", len(cars_data_val))

    testloader = DataLoader(cars_data_test, batch_size=test_batch_size,
                            shuffle=True, num_workers=test_workers, collate_fn=my_collate)
    print("Test data set length:", len(testloader))

    return trainloader, validloader, testloader, cars_data_train.get_num_of_class()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    args = parser.parse_args()
    load_cars(args.data_dir, None, args.validation_split, args.batch_size)
