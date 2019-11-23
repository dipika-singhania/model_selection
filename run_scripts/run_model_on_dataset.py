from __future__ import print_function, division
import os
import sys
import torch
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
from torch.utils.data.dataloader import default_collate
from utils import ValueMeter, topk_accuracy, topk_recall, softmax
import time
import copy
import json

from PIL import Image
from tqdm import tqdm
import sys

from dataset_loaders.image_classification.cars import load_cars
from dataset_loaders.image_classification.pytorch_datasets import load_mnist, load_FashionMNIST, load_KMNIST, \
    load_cifar10, load_cifar100

from model_bin.image_classification.models_pool import ModelsZoo
from model_bin.image_classification.pytorch_models_run import DNN
from model_bin.image_classification.models_pool import MODEL_DICTIONARY
from dataset_loaders.image_classification.data_config import DATA_DICTIONARY



def replace_entry_in_json(fname, newdict):

    old_lines = []
    with open(fname, 'r') as openfile:
        for line in openfile:
            json_obj = json.loads(line.strip())
            old_lines.append(json_obj)

    for count in range(len(old_lines)):
        d_name = json_obj['dataset']
        m_name = json_obj['model_name']
        if d_name == newdict['dataset'] and m_name == newdict['model_name']:
            old_lines[count] = newdict

    with open(fname, 'w') as openfile:
        for json_obj in old_lines:
            json_string = json.dumps(json_obj)
            openfile.write(json_string + '\n')


def train_infer_model_on_dataset(device, model_name, dataset_name, train, evaluate,
                                 resume, overwrite, max_epoch, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optim_param = {'lr': 0.001, 'beta_0': 0.9, 'beta_1': 0.999}
    model_name = 'alexnet'
    model_dump_dir = 'model_pool/'
    final_logs = "logs/model_datasets_meta.json"
    model_file_name = 'model_pool/' + model_name + '_' + dataset_name + '.pth'
    dict_params = None

    dnn = DNN(device, model_name, class_size, batch_size=batch_size, max_epochs=max_epoch, log_file_name=model_name,
              models_save_dir=model_dump_dir, optim_param=optim_param, train_full=True)

    if os.path.exists(model_file_name):
        print('Model file on this dataset already exists!')
        if train and overwrite:
            dnn.train()
            dict_params = dnn.get_dict()
            print('Overwriting model file and rerunning')
        elif resume and train:
            dnn.resume_train()
            dict_params = dnn.get_dict()
            print('Resuming training from saved checkpoint!')
        else:
            dnn.load_parameters()
            if train:
                print('Skipping running model and loading existing one')
                print('To overwrite or resume switch on the respective arguments')
            if evaluate:
                dnn.evaluate()


def train_infer_all_models_in_all_dataset(device, model_list, dataset_list, train, evaluate,
                                          resume, overwrite, max_epoch, batch_size):

    for dataset_name in dataset_list:
        for model in models:
            train_infer_model_on_dataset(device=device, model_name=model, dataset_name=dataset_name, train=train,
                                         evaluate=evaluate, resume=resume, overwrite=overwrite, max_epoch=max_epoch,
                                         batch_size=batch_size)

