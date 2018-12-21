import numpy as np
import collections
import logging
import scipy.signal
import pandas as pd
import os
import re
import json
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def read_log(directory):
    for f in ['log', 'log.json']:
        try:
            with open(os.path.join(directory, f)) as logf:
                js = json.load(logf)
                logg = pd.DataFrame(js)
                logg['directory'] = directory.rstrip('/')
                if 'batch_size' in logg.columns:
                    logg['samples_per_s'] = np.hstack(
                        ([np.nan], np.diff(logg.iteration) *
                         logg.batch_size[1:] / np.diff(logg.elapsed_time)))
                if np.any(logg.epoch == 1):
                    epoch_iterations = logg.iteration.iloc[np.where(
                        logg.epoch == 1)[0][0]]
                    log.info("epoch iterations: %s", epoch_iterations)
                    logg['epochf'] = logg.iteration / epoch_iterations
                return logg
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f)


def read_logs(*result_dirs):
    logs = []
    for result_dir in result_dirs:
        try:
            logg = read_log(result_dir)
            logs.append(logg)
        except FileNotFoundError:
            log.warning("log file not found in directory %s", result_dir)
    ret = pd.concat(logs, sort=False)
    # ret = massage_logs(ret)
    return ret


def smooth(x, half_width=None):
    # default for half_width was 56, it is suited to training loss
    if half_width is None:
        half_width = np.clip(len(x) // 50, 1, None)

    # This is unstable in places (large oscillations up and down).
    # b, a = scipy.signal.butter(2, 0.005)
    # return scipy.signal.filtfilt(b, a, x, method='gust')

    # This is slower than IIR filters, but is more stable.
    width = half_width * 2 + 1
    len_filter = min((len(x) // 3) - 1, width)
    if len_filter <= 1:
        return np.asarray(x)
    log.debug('x size: %d, len filter: %d', len(x), len_filter)
    f = scipy.signal.firwin(len_filter, 0.000005)
    ret = scipy.signal.filtfilt(f, 1, x, padtype='even')

    return ret


def massage_logs(lg):
    lg['elapsed_time'] = pd.to_timedelta(lg.elapsed_time, unit='s')
    cols_train = [c for c in lg.columns if c.startswith('main/')]
    cols_val = [c for c in lg.columns if c.startswith('validation/')]
    cols_others = [
        c for c in lg.columns if c not in cols_train and c not in cols_val
    ]
    cols_metric = sorted(
        set(
            re.sub('^.+/', '', c) for c in lg.columns
            if re.match('^.+/.+$', c) is not None))
    if "samples_per_s" in lg.columns:
        cols_metric += ["samples_per_s"]

    lg_train = lg[cols_others + cols_train].copy()
    lg_train.rename(
        axis='columns', mapper=lambda f: re.sub('^.+/', '', f), inplace=True)
    lg_train['set'] = 'train'

    lg_val = lg[cols_others + cols_val][~pd.isna(lg[cols_val[0]])]
    lg_val.rename(
        axis='columns', mapper=lambda f: re.sub('^.+/', '', f), inplace=True)
    lg_val['set'] = 'val'

    lg = pd.concat((lg_train, lg_val), axis=0, ignore_index=True,
                   sort=False).sort_values(['directory', 'iteration'])

    cols_first = ['directory', 'epoch', 'iteration', 'elapsed_time', 'set']
    cols_last = cols_metric
    cols_middle = sorted(
        [c for c in lg.columns if c not in cols_first and c not in cols_last])
    cols_all = cols_first + cols_middle + cols_last
    cols_all = [x for x in cols_all if x in lg.columns]
    lg = lg[cols_all]

    for metric in cols_metric:
        lg['smooth_' + metric] = lg.groupby(['directory',
                                             'set'])[metric].transform(smooth)

    lg.reset_index(inplace=True, drop=True)
    return lg


def extend_range(r, alpha=0.05):
    mi, ma = r
    delta = (ma - mi)
    mi -= delta * alpha
    ma += delta * alpha
    return (mi, ma)


def compute_ylim(ys, q, raw=False):
    y = np.hstack(ys)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return None
    if raw:
        return extend_range((np.min(y), np.max(y)))
    else:
        r = np.quantile(y, [q, 1. - q])
        for yy in ys:
            r = np.hstack((yy[-10:], r))
        return compute_ylim(r, raw=True, q=q)


def plot(log_directories,
         ys=['loss', 'accuracy', 'lr'],
         x='iteration',
         variants=['train', 'val'],
         y_range='val',
         q_range=0.05,
         figsize=(24, 16)):
    df = massage_logs(read_logs(*log_directories))
    fig, axes = plt.subplots(
        nrows=len(ys), ncols=len(variants), figsize=figsize, sharex=True)
    sets = list(df.set.unique())
    assert set(variants) <= set(sets)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = dict(zip(df.directory.unique(), palette))
    y_range_values = collections.defaultdict(list)
    for (directory, subset), group in df.groupby(['directory', 'set']):
        color = colors[directory]
        xx = group[x]
        if xx.dtype == 'timedelta64[ns]':
            xx = (xx / np.timedelta64(1, 's')) / 3600.
        j = sets.index(subset)
        for i, y in enumerate(ys):
            if y not in group.columns or not np.any(
                    np.isfinite(group[y].values)):
                continue
            ax = axes[i, j]
            smooth_y = "smooth_" + y
            has_smooth = smooth_y in group.columns
            alpha_raw = {False: 0.95, True: 0.2}[has_smooth]
            if has_smooth:
                label_raw = '_nolegend_'
                label_smooth = directory
            else:
                label_raw = directory
                label_smooth = None
            ax.plot(
                xx, group[y], alpha=alpha_raw, label=label_raw, color=color)
            if has_smooth:
                ax.plot(
                    xx,
                    group[smooth_y],
                    alpha=0.95,
                    color=color,
                    label=label_smooth)
            ax.legend()
            ax.set_xlabel(x)
            ax.set_ylabel(subset + " " + y)
            if subset in y_range:
                if has_smooth:
                    y_range_values[i].append(group[smooth_y].values)
                else:
                    y_range_values[i].append(group[y].values)
    for i in range(axes.shape[0]):
        ylim = compute_ylim(y_range_values[i], q=q_range)
        if ylim is not None:
            for ax in axes[i, :]:
                ax.set_ylim(ylim)
    return fig, axes
