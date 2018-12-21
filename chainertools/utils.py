import os
import re
import errno
import hashlib
import logging
import requests
import subprocess
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


CHUNK_SIZE = 32768


# https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
def download_file(url, local_filename):
    if 'drive.google.com' in url:
        file_id = re.search('id=(.+)', url).group(1)
        return download_file_from_google_drive(file_id, local_filename)
    with open(local_filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            for chunk in tqdm(r.iter_content(chunk_size=CHUNK_SIZE)):
                f.write(chunk)


# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination, file_size):
        with open(destination, "wb") as f:
            with tqdm(
                    unit='B', unit_scale=True, unit_divisor=1024,
                    total=file_size) as progress:
                for chunk in response.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    progress.update(CHUNK_SIZE)

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    file_size = -1
    if token:
        params = {'id': file_id, 'confirm': token}

        # The returned header does not include content-length.
        # headers = session.head(url, params=params).headers
        # log.info('headers: %s', headers)
        # file_size = int(headers["Content-Length"])
        # log.info("file size: %d", file_size)

        response = session.get(url, params=params, stream=True)
        # There must be a way to get the content-length, but it eludes
        # me atm.
        # log.info(response.headers)

    save_response_content(response, destination, file_size)


def run(command):
    with subprocess.Popen(
            command, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT) as popen:
        with tqdm(unit='files') as progress:
            for line in iter(popen.stdout.readline, b''):
                progress.update(1)


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
    # tarfile is terribly slow and memory-hungry on a big archive
    # with tarfile.open(tar_file_name) as tar:
    #    tar.extractall(path=destination)
    # tar_ret = subprocess.call([

    run([
        "tar", "--directory", destination, "--verbose", "--totals", "-x", "-f",
        tar_file_name
    ])
    if tar_ret != 0:
        raise RuntimeError("tar returned non-zero: {}".format(tar_ret))
