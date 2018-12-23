__version__ = '0.1.0'

from . import score
from . import train
from . import plot
from . import openimages
from . import utils

import logging
import chainer
import matplotlib.pyplot

chainer.config.autotune = True
chainer.config.cudnn_fast_batch_normalization = True


def init():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    chainer.cuda.get_device_from_id(0).use()
    matplotlib.pyplot.style.use('ggplot')
