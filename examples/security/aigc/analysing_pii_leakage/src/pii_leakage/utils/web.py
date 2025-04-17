# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import os
import tempfile
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from ..global_configs import system_configs


def is_valid_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def generate_hash(url):
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def download_and_unzip(url, extract_to=system_configs.CACHE_DIR):
    hash_name = generate_hash(url)
    path_to_extract = os.path.join(extract_to, hash_name)

    if os.path.exists(path_to_extract):
        print(f"The file from '{url}' has already been downloaded to '{os.path.abspath(path_to_extract)}'.")
        return

    os.makedirs(path_to_extract, exist_ok=True)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))

    progress = tqdm(response.iter_content(1024), f'Downloading {url}', total=file_size, unit='B', unit_scale=True,
                    unit_divisor=1024)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in progress.iterable:
            tmp_file.write(chunk)
            progress.update(len(chunk))

    with zipfile.ZipFile(tmp_file.name, 'r') as thezip:
        for zip_info in thezip.infolist():
            if zip_info.filename[-1] == '/':
                continue  # skip directories
            zip_info.filename = os.path.basename(zip_info.filename)  # strip the path
            thezip.extract(zip_info, path_to_extract)

    os.unlink(tmp_file.name)  # remove the temporary file
    print(f"Downloaded and extracted '{url}' to '{os.path.abspath(path_to_extract)}'")
    return path_to_extract
