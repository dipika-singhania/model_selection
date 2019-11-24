import requests
from tqdm import tqdm
import os
import tempfile
import dataset_loaders.image_classification.data_config as data_config


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
