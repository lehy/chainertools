import chainer
import copy
import glob
import logging
import gc
import os
import functools
import pandas as pd
from . import utils

log = logging.getLogger(__name__)


class OnlineEvaluator(chainer.training.extensions.Evaluator):
    """An evaluator that evaluates one batch at a time.
    """

    def __init__(self, *args, **kwargs):
        super(OnlineEvaluator, self).__init__(*args, **kwargs)

    @staticmethod
    def _reset_iterator(iterator):
        if hasattr(iterator, 'reset'):
            iterator.reset()
            return iterator
        else:
            return copy.copy(iterator)

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        if self.eval_hook:
            self.eval_hook(self)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self._reset_iterator(iterator)
            batch = next(iterator)

        summary = chainer.reporter.DictSummary()

        observation = {}
        with chainer.reporter.report_scope(observation):
            in_arrays = self.converter(batch, self.device)
            with chainer.function.no_backprop_mode():
                if isinstance(in_arrays, tuple):
                    eval_func(*in_arrays)
                elif isinstance(in_arrays, dict):
                    eval_func(**in_arrays)
                else:
                    eval_func(in_arrays)

        summary.add(observation)

        return summary.compute_mean()


# Note that there is a TimeTrigger in chainer, and it probably does
# exactly this.
class TimeTrigger(object):
    def __init__(self, period_s):
        self.period_s = period_s
        self.t_last_triggered = None

    def __call__(self, trainer):
        t = trainer.elapsed_time
        if self.t_last_triggered is None or (
                t - self.t_last_triggered) > self.period_s:
            self.t_last_triggered = t
            return True
        else:
            return False

    def serialize(self, serializer):
        self.period_s = serializer('period_s', self.period_s)
        self.t_last_triggered = serializer('t_last_triggered',
                                           self.t_last_triggered)


def trigger_every_iteration_until(max_iterations=256):
    def f(trainer):
        iteration = trainer.updater.iteration
        if iteration < max_iterations:
            return True
        else:
            if iteration == max_iterations:
                log.info("reached maximum iterations for printing: %d",
                         max_iterations)
            return False

    return f


def extend_log_print(trainer):
    optimizer = trainer.updater.get_optimizer('main')

    hyperparams = list(optimizer.hyperparam.get_dict().keys())

    for hyperparam in hyperparams:

        def getter(trainer, h=hyperparam):
            return getattr(optimizer, h)

        trainer.extend(
            chainer.training.extensions.observe_value(hyperparam, getter),
            trigger=(1, 'iteration'))

    trainer.extend(
        chainer.training.extensions.observe_value(
            'finetune',
            lambda _trainer: getattr(chainer.config, 'finetune', False)),
        trigger=(1, 'iteration'))

    def get_batch_size(trainer):
        return trainer.updater.get_iterator('main').batch_size

    trainer.extend(
        chainer.training.extensions.observe_value('batch_size',
                                                  get_batch_size),
        trigger=(1, 'iteration'))

    # This sends observations to the log. The trigger passed to the
    # LogReport init is how often the log is written. Observations are
    # still collected every iteration (which is the default trigger
    # used by extend()).
    trainer.extend(
        chainer.training.extensions.LogReport(trigger=(20, 'iteration')))

    # This prints observations to the console.
    observed_variables = [
        'elapsed_time',
        'epoch',
        'iteration',
        'main/loss',
        'validation/main/loss',
        'finetune',
        'batch_size',
    ] + hyperparams
    model = trainer.updater.get_optimizer('main').target
    if getattr(model, 'compute_accuracy', False):
        observed_variables += ['main/accuracy', 'validation/main/accuracy']

    # There is not much sense setting this to a TimeTrigger, since it
    # displays all logged entries since the last time, not one every
    # time it wakes up.  TODO: wake up a few times at the beginning
    # every n iterations (or n seconds), then once every epoch (or
    # every 30 mn).
    trainer.extend(
        chainer.training.extensions.PrintReport(observed_variables),
        trigger=trigger_every_iteration_until(128))
    # trigger=(1, 'iteration'))


def extend_evaluation(trainer, iterator, validation_dataset,
                      validation_batch_size, evaluation_factor):
    train_batch_size = trainer.updater.get_iterator('main').batch_size
    validation_iter = iterator(
        validation_dataset,
        batch_size=validation_batch_size,
        repeat=False,
        shuffle=False)
    trigger_evaluation = (max(
        int(validation_batch_size / (train_batch_size * evaluation_factor)),
        1), 'iteration')
    model = trainer.updater.get_optimizer('main').target
    device = getattr(model, '_device_id', None)
    trainer.extend(
        OnlineEvaluator(validation_iter, model, device=device),
        trigger=trigger_evaluation)


class SnapshotOnFinalize(chainer.training.Extension):
    """Extension to take a snapshot when training ends.

    This triggers even when training is interrupted with C-c.
    """

    def __init__(self):
        self.trainer = None

    def initialize(self, trainer):
        log.info("setting up trainer snapshot on finalization")
        self.trainer = trainer
        gc.collect(
        )  # not a logical place, but potentially useful and harmless

    def __call__(self, trainer):
        self.trainer = trainer

    def finalize(self):
        """XXX TODO check that the written snapshot is reasonable (ie it
        includes a full model).
        """
        log.info("saving final trainer state")
        # Save with a different prefix than regular snapshots, to
        # avoid clobbering one of them. We might also want to avoid
        # snapshotting if the model is not initialized.
        chainer.training.extensions.snapshot(
            filename="f_snapshot_iter_{.updater.iteration}")(self.trainer)


def extend_snapshot(trainer):
    # this saves the complete trainer object
    trainer.extend(
        chainer.training.extensions.snapshot(), trigger=TimeTrigger(30 * 60))
    trainer.extend(SnapshotOnFinalize(), trigger=lambda _trainer: False)


def resume_from_snapshot(trainer, snapshot=None):
    gc.collect()
    # Run one dummy update to make sure all parameters that need to be
    # reloaded are created. The reporter thing is the minimal thing
    # that seems necessary for update() to run without problems. (This
    # is copied from the Trainer code.)
    reporter = chainer.reporter.Reporter()
    optimizer = trainer.updater.get_optimizer('main')
    reporter.add_observer('main', optimizer.target)
    reporter.add_observers('main', optimizer.target.namedlinks(skipself=True))
    with reporter.scope({}):
        trainer.updater.update()

    # Using "snapshot_iter*" to NOT get the automatic interruption
    # snapshots, which are for the moment sometimes empty if we have a
    # crash before optim starts. The real fix would be to ensure that
    # finalize() snapshots are always valid when writing them.
    snapshots = glob.glob(os.path.join(trainer.out, "snapshot_iter_*"))
    if not snapshots:
        log.info("did not find snapshot to resume from in directory %s",
                 trainer.out)
        return
    sorted_snapshots = pd.Series(snapshots).str.extract(
        r'^(?P<file>.*/[^/]*snapshot_iter_0*(?P<iteration>\d+))$')
    sorted_snapshots['iteration'] = sorted_snapshots.iteration.astype(int)
    sorted_snapshots = sorted_snapshots.sort_values("iteration").reset_index(
        drop=True)
    if sorted_snapshots.iteration.iloc[-1] == 0:
        return
    log.info("latest available snapshots:\n%s", sorted_snapshots.tail(7))
    if snapshot is None:
        snapshot = sorted_snapshots.file.iloc[-1]
    log.info("resuming from snapshot %s", snapshot)
    chainer.serializers.load_npz(snapshot, trainer)


def create_iterator(name_or_iterator):
    d = dict(
        serial=chainer.iterators.SerialIterator,
        multiprocess=chainer.iterators.MultiprocessIterator,
        multithread=functools.partial(
            chainer.iterators.MultithreadIterator, n_threads=8))
    if name_or_iterator in d:
        return d[name_or_iterator]
    else:
        log.info("iterator is not in %s, using iterator as passed: %s",
                 list(d.keys()), name_or_iterator)
        return name_or_iterator


def create_model_with_loss(predictor, classifier, loss_fun, accuracy_fun):
    if classifier is None:
        log.info(
            "classifier is None, assuming predictor already outputs a loss")
        return predictor

    kwargs = {}
    if loss_fun is not None:
        kwargs['lossfun'] = loss_fun

    if accuracy_fun is not None:
        kwargs['accfun'] = accuracy_fun

    model = classifier(predictor, **kwargs)
    if 'accfun' not in kwargs:
        log.info('accuracy_fun not passed, disabling accuracy')
        model.compute_accuracy = False

    return model


def train(predictor,
          data,
          name,
          train_batch_size=128,
          validation_batch_size=None,
          optimizer=None,
          iterator='multithread',
          classifier=chainer.links.Classifier,
          evaluation_factor=0.1,
          loss_fun=None,
          accuracy_fun=None,
          device=0):
    """evaluation_factor: evaluate this many samples for each sample
    trained
    """
    gc.collect()
    iterator = create_iterator(iterator)

    model = create_model_with_loss(predictor, classifier, loss_fun,
                                   accuracy_fun)
    model.to_gpu(device=device)

    if optimizer is None:
        log.warning("no optimizer given, using default Adam (no weight decay)")
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = iterator(
        data.train, batch_size=train_batch_size, shuffle=True)
    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)

    name = "{}-{}-batch{}".format(name, predictor.__class__.__name__,
                                  train_batch_size)
    log.info("saving model as: %s", name)
    trainer = chainer.training.Trainer(updater, out=name)

    if validation_batch_size is None:
        validation_batch_size = train_batch_size
    extend_evaluation(trainer, iterator, data.validation,
                      validation_batch_size, evaluation_factor)

    extend_log_print(trainer)
    extend_snapshot(trainer)

    return trainer


class UpDownLr(chainer.training.Extension):
    """2 epochs going up from max_lr/10 to max_lr, then plateaus of
    constant lr, reduced by 0.3 every 10 epochs.  finetune mode after
    4 cycles of 10 epochs.  Optimizer is reset using
    optim.setup(target) at the beginning of each cycle.
    """

    def __init__(self,
                 max_lr,
                 factor_min_lr=0.1,
                 num_epochs_to_max=2,
                 num_epochs_reduce=10,
                 reduction=0.3,
                 cycles_before_finetune=-1,
                 lr_attribute=None):
        super().__init__()
        self.max_lr = max_lr
        self.factor_min_lr = factor_min_lr
        self.num_epochs_to_max = num_epochs_to_max
        self.num_epochs_reduce = num_epochs_reduce
        self.reduction = reduction
        self.lr_attribute = lr_attribute
        self.cycles_before_finetune = cycles_before_finetune
        self.base_iteration = 0
        self.info_emitted = False

    def initialize(self, trainer):
        # this is called also after deserializing, keep the current state!
        self(trainer)

    def __call__(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        train_iter = trainer.updater.get_iterator('main')

        iteration = self.base_iteration + optimizer.t
        size_epoch_samples = len(train_iter.dataset)
        batch_size = train_iter.batch_size
        size_epoch_iterations = size_epoch_samples / batch_size

        iterations_go_up = self.num_epochs_to_max * size_epoch_iterations
        iterations_constant = self.num_epochs_reduce * size_epoch_iterations
        iterations_before_finetune = (
            self.cycles_before_finetune * self.num_epochs_reduce *
            size_epoch_iterations)

        min_lr = self.factor_min_lr * self.max_lr

        if not self.info_emitted:
            log.info("epoch: %d samples, %d iterations (batch size: %d)",
                     size_epoch_samples, size_epoch_iterations, batch_size)
            log.info(
                "initial linear scale up of lr " +
                "from %g to %g in %g epochs (%g iterations)", min_lr,
                self.max_lr, self.num_epochs_to_max, iterations_go_up)
            log.info(
                "then lr has plateaus of %g epochs (%g iterations)" +
                ", reduced each time by %g", self.num_epochs_reduce,
                iterations_constant, self.reduction)
            self.info_emitted = True

        if iteration < iterations_go_up:
            lr = min_lr + (
                self.max_lr - min_lr) * iteration / float(iterations_go_up)
        else:
            cycle_index = (iteration - iterations_go_up) // iterations_constant
            lr = self.max_lr * (self.reduction**cycle_index)

        finetune = (iterations_before_finetune > 0 and (
            (iteration - iterations_go_up) > iterations_before_finetune))

        if self.lr_attribute is None:
            if isinstance(optimizer, chainer.optimizers.Adam):
                log.debug("detected Adam optimizer, adapting eta")
                self.lr_attribute = 'eta'
            elif utils.safe_hasattr(optimizer, 'lr'):
                log.debug("detected optimizer with lr attribute, adapting lr")
                self.lr_attribute = 'lr'
            else:
                raise ValueError("optimizer has neither lr nor alpha" +
                                 ", you must specify lr_attribute at init")

        # Reset the optimizer (Adam state for instance) if at the
        # beginning of a cycle (not the first one though).
        if (iteration > iterations_go_up and lr != self.current_lr(optimizer)):
            log.info(
                "beginning new cycle with %s = %g (current: %g)," +
                " resetting optimizer", self.lr_attribute, lr,
                self.current_lr(optimizer))
            # Calling setup() resets optimizer.t to zero. Make sure we
            # remember the correct number of iterations done until
            # this point.
            self.base_iteration = iteration
            optimizer.setup(optimizer.target)

        self.set_lr(optimizer, lr)
        if getattr(chainer.config, 'finetune', None) != finetune:
            log.info("setting finetune=%s", finetune)
            chainer.config.finetune = finetune

    def current_lr(self, optimizer):
        return getattr(optimizer, self.lr_attribute)

    def set_lr(self, optimizer, lr):
        setattr(optimizer, self.lr_attribute, lr)

    def serialize(self, serializer):
        self.max_lr = serializer('max_lr', self.max_lr)
        self.factor_min_lr = serializer('factor_min_lr', self.factor_min_lr)
        self.num_epochs_to_max = serializer('num_epochs_to_max',
                                            self.num_epochs_to_max)
        self.num_epochs_reduce = serializer('num_epochs_reduce',
                                            self.num_epochs_reduce)
        self.reduction = serializer('reduction', self.reduction)
        self.cycles_before_finetune = serializer('cycles_before_finetune',
                                                 self.cycles_before_finetune)
        # Need str() because otherwise we get an ndarray of chars and
        # getattr does not like that.
        self.lr_attribute = str(serializer('lr_attribute', self.lr_attribute))
        self.base_iteration = serializer('base_iteration', self.base_iteration)
