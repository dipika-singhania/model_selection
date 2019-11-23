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

from dataset_loaders.image_classification.flower import load_flower

from model_bin.image_classification.models_pool import ModelsZoo
from model_bin.image_classification.pytorch_models_run import DNN


def run_flower_on_all_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optim_param = {'lr': 0.001, 'beta_0': 0.9, 'beta_1':0.999}
    model_dump_dir = 'model_pool/'
    final_logs = "logs/model_datasets_meta.json"

    train_loader,valid_loader, test_loader, class_size = load_flower('dataset_pool/', batch_size=64)
    model_name = 'resnet34'
    dnn = DNN(device, model_name, batch_size=64, max_epochs=20, log_file_name=model_name, models_save_dir=model_dump_dir, optim_param=optim_param, train_full=True)
    dnn.train(train_loader, valid_loader)
    dict_params = dnn.get_dict()
    dict_params['dataset'] = 'flower'
    with open(final_logs, 'a+') as fp:
        json_string = json.dumps(dict_params)
        fp.write(json_string + "\n")

    train_loader,valid_loader, test_loader, class_size = load_flower('dataset_pool/', batch_size=120)
    model_name = 'mobilenet_v2'
    dnn = DNN(device, model_name, batch_size=120, max_epochs=20, log_file_name=model_name, models_save_dir=model_dump_dir, optim_param=optim_param, train_full=True)
    dnn.train(train_loader, valid_loader)
    dict_params = dnn.get_dict()
    dict_params['dataset'] = 'flower'
    with open(final_logs, 'a+') as fp:
        json_string = json.dumps(dict_params)
        fp.write(json_string + "\n")

    train_loader,valid_loader, test_loader, class_size = load_flower('dataset_pool/', batch_size=64)
    model_name = 'vgg16'
    dnn = DNN(device, model_name, batch_size=64, max_epochs=20, log_file_name=model_name, models_save_dir=model_dump_dir, optim_param=optim_param, train_full=True)
    dnn.train(train_loader, valid_loader)
    dict_params = dnn.get_dict()
    dict_params['dataset'] = 'flower'
    with open(final_logs, 'a+') as fp:
        json_string = json.dumps(dict_params)
        fp.write(json_string + "\n")


if __name__ == '__main__':
    run_flower_on_all_models()
    
