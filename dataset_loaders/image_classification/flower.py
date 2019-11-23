import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from meta_model_bin.get_labels_to_vec import get_word_list_embeddings, add_vector_to_file
import glob


class Flower(Dataset):
    def __init__(self, root_dir, train=False, transform=None, validation_split=0.3):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        base_folder = "flowers"
        directory = os.path.join(root_dir, base_folder)
        self.transform = transform
        self.base_folder = base_folder
        self.create_csv(directory, train, validation_split)

    def create_csv(self, directory, train, validation_split):
        full_list = []
        labels_dic = dict()
        count = 0
        for direc in glob.glob(directory + "/*/*.jpg"):
             name = os.path.basename(direc)
             direc_label = os.path.dirname(direc)
             label = os.path.basename(direc_label)
             if label not in labels_dic:
                labels_dic[label] = count
                count += 1
             full_list.append((direc, labels_dic[label]))
       
        np.random.shuffle(full_list)
        full_lenght = len(full_list)
        if train:
            self.final_list = full_list[: int(-1 * validation_split * full_lenght)]
        else:
            self.final_list = full_list[int(-1 * validation_split * full_lenght):]

    def get_num_of_classes(self):
        return 6

    def get_features(self):
        img_path, classid = self.final_list[0]
        img_height, img_width, _ = np.array(Image.open(img_path)).shape
        total_train_size = len(self.final_list)
        num_of_classes = self.get_num_of_classes()
        dataset_feature = get_word_list_embeddings('flower', ['daisy','rose','sunflower','tuplip','flowers','dandelion']) 
        add_vector_to_file('flower', img_height, img_width, total_train_size, num_of_classes, dataset_feature)

    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, idx):
        img_path, classId = self.final_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


def load_flower(data_dir, transform=None, validation_split=0.2, batch_size=256, create_features=False):
    train_batch_size = batch_size
    test_batch_size = batch_size
    train_workers = 4
    test_workers = 4

    if transform is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
                                           transforms.Scale((224, 224)),
                                           transforms.ToTensor(),
                                           normalize, 
                                       ])
    train_dataset = Flower(data_dir, True, transform, validation_split)
    valid_dataset = Flower(data_dir, False, transform, validation_split)

    get_ele = train_dataset.__getitem__(1)
    if create_features is True:
        train_dataset.get_features()

    trainloader = DataLoader(train_dataset, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers)  #, collate_fn=my_collate)
    print("Train data set length:", len(train_dataset))

    validloader = DataLoader(valid_dataset, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers)  #, collate_fn=my_collate)
    print("Validation data set length:", len(valid_dataset))

    testloader = DataLoader(valid_dataset, batch_size=test_batch_size,
                            shuffle=True, num_workers=test_workers)  #, collate_fn=my_collate)
    print("Test data set length:", len(valid_dataset))

    return {'train_loader': trainloader, 'validation_loader': validloader,
            'test_loader': testloader, 'num_classes': train_dataset.get_num_of_classes(), 'name': 'flower'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_pool/')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    args = parser.parse_args()
    # load_flower(args.data_dir, None, args.validation_split, args.batch_size, True)

