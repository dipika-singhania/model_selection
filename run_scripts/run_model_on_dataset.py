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
from model_bin.utils import ValueMeter, topk_accuracy, topk_recall
import time
import copy
import json

from PIL import Image
from tqdm import tqdm
import sys

from model_bin.image_classification.models_pool import ModelsZoo
from model_bin.image_classification.pytorch_models_run import DNN
from model_bin.image_classification.models_pool import MODEL_DICTIONARY
from dataset_loaders.image_classification.data_config import DATA_DICTIONARY, return_dataloader


def replace_entry_in_json(fname, newdict):
    old_lines = []
    with open(fname, 'r') as openfile:
        for line in openfile:
            json_obj = json.loads(line.strip())
            old_lines.append(json_obj)

    for count in range(len(old_lines)):
        d_name = old_lines[count]['dataset']
        m_name = old_lines[count]['model_name']
        if d_name == newdict['dataset'] and m_name == newdict['model_name']:
            old_lines[count] = newdict

    with open(fname, 'w') as openfile:
        for json_obj in old_lines:
            json_string = json.dumps(json_obj)
            openfile.write(json_string + '\n')


def train_infer_model_on_dataset(device, model_name, dataset_name, train, evaluate, train_full, data_dir,
                                 resume, overwrite, max_epoch, batch_size, final_logs, model_dump_dir):
    optim_param = {'lr': 0.001, 'beta_0': 0.9, 'beta_1': 0.999}
    model_file_name = 'model_pool/' + model_name + '_' + dataset_name + '.pth.tar'
    dict_params = None

    print('Running %s model on %s dataset' %(model_name, dataset_name))

    dataset_dict = return_dataloader(data_name=dataset_name, root=data_dir, batch_size=batch_size)
    dnn = DNN(device=device, model_name=model_name, batch_size=batch_size, max_epochs=max_epoch, dataset_dict=dataset_dict,
              models_save_dir=model_dump_dir, optim_param=optim_param, train_full=train_full)

    if train and not os.path.exists(model_file_name):
        print('Training from scratch as model file doesnt exist!')
        dnn.train()
        dict_params = dnn.get_dict()
    elif train and overwrite:
        print('Overwriting model file and rerunning')
        dnn.train()
        dict_params = dnn.get_dict()
    elif resume and train:
        print('Resuming training from saved checkpoint!')
        dnn.resume_train()
        dict_params = dnn.get_dict()
    else:
        dnn.load_parameters()
        if train:
            print('Skipping running model and loading existing one')
            print('To overwrite or resume switch on the respective arguments')
        if evaluate:
            print('Running inference only!')
            dnn.evaluate(dataset_dict['validation_loader'])
    if dict_params is not None:
        replace_entry_in_json(final_logs, dict_params)


def train_infer_all_models_in_all_dataset(device, model_list, dataset_list, train, evaluate, final_logs,
                                          resume, overwrite, max_epoch, batch_size, model_dump_dir, data_dir):
    for dataset_name in dataset_list:
        for model in model_list:
            train_full = not MODEL_DICTIONARY[model]['pretrained']
            train_infer_model_on_dataset(device=device, model_name=model, dataset_name=dataset_name, train=train,
                                         evaluate=evaluate, resume=resume, overwrite=overwrite, max_epoch=max_epoch,
                                         batch_size=batch_size, model_dump_dir=model_dump_dir, final_logs=final_logs,
                                         train_full=train_full, data_dir=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_log', type=str, default='logs/model_datasets_meta.json', help="Final log storage")
    parser.add_argument('--model_save_dir', type=str, help="Model save dir", default="model_pool/")
    parser.add_argument('--dataset_dir', type=str, help="Dataset dir", default="dataset_pool/")
    parser.add_argument('--model_name', type=str,
                        choices=list(MODEL_DICTIONARY.keys()) + ['_all_'], help="Pytorch model name to run",
                        default="resnet18")
    parser.add_argument('--resume', action='store_true', help="Whether to resume training")
    parser.add_argument('--overwrite', action='store_true', help="Whether to run overwrite")
    parser.add_argument('--train', action='store_true', help="Whether to run train")
    parser.add_argument('--infer', action='store_true', help="Whether to run inference")
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    parser.add_argument('--max_epoch', type=int, default=100, help="max epochs to run a model")
    parser.add_argument('--dataset', choices=list(DATA_DICTIONARY.keys()) + ['_all_'], help="Name of the dataset")
    parser.add_argument('--data_dir', default='dataset_pool/', help="Data location")
    parser.add_argument('--device', default=None, help="cpu / gpu")

    (args, _) = parser.parse_known_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if args.dataset == '_all_':
        dataset_list = list(DATA_DICTIONARY.keys())
    else:
        if args.dataset is None:
            print("Setting to pick up default dataset cifar10")
            args.dataset = 'cifar10'
        dataset_list = [args.dataset]

    if args.model_name == '_all_':
        model_list = list(MODEL_DICTIONARY.keys())
    else:
        model_list = [args.model_name]

    if args.infer is False and args.train is False:
        print("Setting to inference mode")
        args.infer = True

    train_infer_all_models_in_all_dataset(device=device, model_list=model_list, dataset_list=dataset_list,
                                          train=args.train, evaluate=args.infer, resume=args.resume,
                                          overwrite=args.overwrite, max_epoch=args.max_epoch, batch_size=args.batch_size,
                                          model_dump_dir=args.model_save_dir, final_logs=args.final_log,
                                          data_dir=args.dataset_dir)
