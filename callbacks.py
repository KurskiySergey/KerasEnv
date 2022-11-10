import numpy
from keras.callbacks import Callback, ModelCheckpoint
from config import KERAS_DIR
import os


model_checkpoint = ModelCheckpoint(
    filepath="./checkpoint",
    monitor="loss",
    mode="min",
    save_best_only=True
)


class KerasCallback(Callback):

    def __init__(self, filepath="callback_model", model_to_save=None, use_callback=True):
        super().__init__()
        self.use_callback = use_callback
        self.loss = numpy.Inf
        self.filepath = os.path.join(KERAS_DIR, filepath)
        self.model_to_save = model_to_save

    def on_train_begin(self, logs=None):
        self.loss = numpy.Inf

    def on_epoch_end(self, batch, logs=None):
        if self.use_callback:
            if logs.get("loss") < self.loss:
                self.loss = logs.get("loss")
                print()
                self.model_to_save.save_model(self.filepath)
