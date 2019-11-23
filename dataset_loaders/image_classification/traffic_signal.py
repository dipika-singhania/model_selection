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


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def get_num_of_classes(self):
        return 43

    def get_features(self):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[0, 0])
        img_height, img_width, _ = np.array(Image.open(img_path)).shape
        total_train_size = len(self.csv_data)
        num_of_classes = self.get_num_of_classes()
        dataset_feature = get_word_list_embeddings('traffic', ['stop','turn right','turn left']) 
        add_vector_to_file('traffic', img_height, img_width, total_train_size, num_of_classes, dataset_feature)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


def load_traffic(data_dir, transform=None, validation_split = 0.2, batch_size=256, require_features=False):
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
    train_dataset = GTSRB(data_dir, True, transform)
    valid_dataset = GTSRB(data_dir, False, transform)

    get_ele = train_dataset.__getitem__(1)
    if require_features is True:
        train_dataset.get_features()

    trainloader = DataLoader(train_dataset, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers) #, collate_fn=my_collate)
    print("Train data set length:", len(train_dataset))

    validloader = DataLoader(valid_dataset, batch_size=train_batch_size,
                             shuffle=True, num_workers=train_workers) #, collate_fn=my_collate)
    print("Validation data set length:", len(valid_dataset))

    testloader = DataLoader(valid_dataset, batch_size=test_batch_size,
                            shuffle=True, num_workers=test_workers) #, collate_fn=my_collate)
    print("Test data set length:", len(valid_dataset))

    return {'train_loader': trainloader, 'validation_loader': validloader,
            'test_loader': testloader, 'num_classes': train_dataset.get_num_of_classes(), 'name': 'traffic'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset_pool/')
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=float, default=64, help="batch_size of model")
    args = parser.parse_args()
    # load_traffic(args.data_dir, None, args.validation_split, args.batch_size, True)
