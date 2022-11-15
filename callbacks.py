import numpy
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from config import KERAS_DIR
import os


model_checkpoint = ModelCheckpoint(
    filepath="./checkpoint",
    monitor="loss",
    mode="min",
    save_best_only=True
)

EPOCH_STEP = 1000

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


def scheduler(epoch, lr):
    updated_lr = lr
    if epoch+1 - ((epoch + 1) // EPOCH_STEP) * EPOCH_STEP == 0:
        updated_lr /= 10
    print(f"lr scheduler epoch {epoch}: {updated_lr}")
    return updated_lr


