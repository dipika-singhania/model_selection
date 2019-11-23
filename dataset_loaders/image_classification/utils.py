import requests
from tqdm import tqdm
import os
import tempfile
import data_config

def download_dataset_from_url(dataset_url):
    '''
        Download the dataset at URL over HTTP/HTTPS, ensuring that the dataset ends up in the local filesystem.
        Shows a progress bar.

        :param str dataset_url: Publicly available URL to download the dataset
        :returns: path of the dataset in the local filesystem (should be deleted after use)
    '''
    print('Downloading dataset from {dataset_url}...')

    r = requests.get(dataset_url, stream=True)

    # Show a progress bar while downloading
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    iters = math.ceil(total_size / block_size)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        for data in tqdm(r.iter_content(block_size), total=iters, unit='KB'):
            f.write(data)

        return f.name

def return_dataloader(data_name, root, batch_size):
    '''
    :param data_name: Name of the dataset
    :return: a dictionary with keys 'name', 'train', 'valid', 'test'
    '''
    data_dict = data_config.DATA_DICTIONARY
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
