import os
import re
import errno
import requests
import tarfile
import hashlib
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __reduce__(self):
        return (dict, (dict(self), ))


def safe_hasattr(o, k):
    try:
        return hasattr(o, k)
    except:
        return False


# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    """Ensure directory `path` exists.

    Like `mkdir -p`, all needed subdirectories in `path` are created
    if necessary.

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
def download_file(url, local_filename):
    if 'drive.google.com' in url:
        file_id = re.search('id=(.+)', url).group(1)
        return download_file_from_google_drive(file_id, local_filename)
    with open(local_filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)


# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def download_tar(url, destination):
    mkdir_p(destination)

    temp_tar_file_name = os.path.join(
        destination,
        hashlib.md5(url.encode('utf-8')).hexdigest())
    tar_file_name = temp_tar_file_name + '.tar'

    if not os.path.exists(tar_file_name):
        log.info("downloading %s to %s", url, tar_file_name)
        download_file(url, temp_tar_file_name)
        os.rename(temp_tar_file_name, tar_file_name)

    log.info("extracting %s to directory %s", tar_file_name, destination)
    with tarfile.open(tar_file_name) as tar:
        tar.extractall(path=destination)
