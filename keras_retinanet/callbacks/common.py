import keras.callbacks
import numpy as np
import time
from .. import losses
from ..utils.model import unfreeze as unfreeze_model
from ..utils.model import unfreeze_from_block as unfreeze_model_from_block


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        self.times.append(epoch_time)
        print("Epoch time: {}".format(epoch_time))

    def on_train_end(self, logs=None):
        print("Average epoch time: {}".format(np.average(self.times)))


class ScheduledFreeze(keras.callbacks.Callback):
    def __init__(self, model, schedule):
        super(ScheduledFreeze, self).__init__()

        self.model = model
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.schedule:
            block_id = self.schedule[epoch]
            if block_id is None:
                self.model = unfreeze_model(self.model)
            else:
                self.model = unfreeze_model_from_block(self.model, block_id)

            self.model.compile(
                loss={
                    'regression': losses.smooth_l1(),
                    'classification': losses.focal()
                },
                optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
            )
            print(self.model.summary())


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
