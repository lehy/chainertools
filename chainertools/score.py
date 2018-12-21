import chainer
import numpy as np


def f1_score(y, t, threshold_pre_sigmoid=0.):
    xp = chainer.backend.get_array_module(y, t)

    # So it seems that y comes as a Variable, and t comes as an array
    # (this makes sense once you say it!).
    y = y.array

    pred = (xp.ravel(y) > threshold_pre_sigmoid).astype(np.int32)
    t = xp.ravel(t)

    relevant = xp.sum(pred)
    support = xp.sum(t)
    tp = xp.sum((t == 1) & (pred == 1))
    precision = tp / (relevant + 1e-5)
    recall = tp / (support + 1e-5)

    f1 = 2. * precision * recall / (precision + recall + 1e-5)

    # log.info("tp=%d relevant=%d support=%d", tp, relevant, support)
    # log.info("precision=%g recall=%g f1=%g", precision, recall, f1)

    if isinstance(f1, (np.float, np.float16, np.float32, np.float64)):
        f1 = np.array([f1])

    return chainer.Variable(f1)
