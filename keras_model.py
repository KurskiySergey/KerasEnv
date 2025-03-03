import json

import keras
from datasets import Dataset
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self, print_predict=True):
        self.model: keras.Model = None
        self.dataset: Dataset = None
        self.testX, self.testY = None, None
        self.trainX, self.trainY = None, None
        self.valX, self.valY = None, None
        self.prediction = None
        self.history = None
        self.callbacks = []
        self.print_predict=print_predict

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def save_prediction(self):
        pass

    def load_dataset(self, fit_generator=False, use_batches=True, split_generator=False):
        self.dataset.load_dataset(fit_generator=fit_generator, use_batches=use_batches, split_generator=split_generator)
        self.testX, self.testY = self.dataset.get_test_data()
        self.trainX, self.trainY = self.dataset.get_train_data()
        self.valX, self.valY = self.dataset.get_val_data()

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def load_model(self, model_name, custom_objects=None):
        print(f"loading model_efficientnet from {model_name}...")
        self.model = keras.models.load_model(model_name, custom_objects=custom_objects)
        print("done")
        print(self.model)

    def save_model(self, model_name):
        print(f"saving model_efficientnet to {model_name}")
        self.model.save(model_name)
        print("done")

    def save_history(self, file_name):
        print(f"saving model history to {file_name}")
        try:
            with open(f"{file_name}_history.json", "w") as history:
                model_history = self.history.history
                lr_data = model_history.get("lr")
                lr_data = [float(data) for data in lr_data]
                model_history["lr"] = lr_data
                json.dump(model_history, history)
        except (json.JSONDecodeError, TypeError):
            with open(f"{file_name}_history.txt", "w") as history:
                history.write(str(self.history.history))
        print("done")

    def test(self):
        score = self.model.evaluate(self.testX, self.testY, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def compile(self, loss, optimizer, metrics, run_eagerly=False):
        print("compiling model_efficientnet...")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)
        print("done")

    def train(self, batch_size=128, epochs=15, **kwargs):
        no_callback = kwargs.get('no_callback')
        kwargs.pop('no_callback')

        if no_callback:
            callbacks = []
        else:
            callbacks = self.callbacks

        if isinstance(self.model, keras.Model):
            print("training model_efficientnet ...")
            print(f"batch-size = {batch_size}, epochs = {epochs}")
            try:
                print(f"train X len = {len(self.trainX)} test X len = {len(self.testX)}")
            except TypeError:
                print(f"train X is generator")
            if self.trainY is None:
                self.history = self.model.fit(self.trainX,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              callbacks=callbacks,
                                              **kwargs)
            else:
                self.history = self.model.fit(self.trainX, self.trainY,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              callbacks=callbacks,
                                              **kwargs)
            print("done")
        else:
            print("can`t start training. Model is not set")

    def predict(self, X: list, Y: list = None):
        transform_X = np.asarray([self.dataset.parser.file_data_transform(X_el) for X_el in X])
        transform_X = self.dataset.parser.input_transformer(transform_X)

        prediction = self.dataset.parser.prediction_transform(self.model.predict(transform_X))
        if self.print_predict:
            print("Prediction of input data:")
            print(prediction)
            if Y is not None:
                print("Real output")
                print(Y)

        self.prediction = prediction

    def __repr__(self):
        print(self.model.summary())
        print("...")
        print("layers info: ")
        for layer in self.model.layers:
            print(layer.get_config(), layer.get_weights())

        return "end"

    def plot_model(self):
        keras.utils.plot_model(self.model, show_shapes=True)
