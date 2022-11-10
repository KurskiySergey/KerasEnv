import keras
from datasets import Dataset
from callbacks import KerasCallback, model_checkpoint
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self):
        self.model: keras.Model = None
        self.dataset: Dataset = None
        self.testX, self.testY = None, None
        self.trainX, self.trainY = None, None
        self.prediction = None
        self.history = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def save_prediction(self):
        pass

    def load_dataset(self):
        self.dataset.load_dataset()
        self.testX, self.testY = self.dataset.get_test_data()
        self.trainX, self.trainY = self.dataset.get_train_data()

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def load_model(self, model_name):
        print(f"loading model_efficientnet from {model_name}...")
        self.model = keras.models.load_model(model_name)
        print("done")
        print(self.model)

    def save_model(self, model_name):
        print(f"saving model_efficientnet to {model_name}")
        self.model.save(model_name)
        print("done")

    def test(self):
        score = self.model.evaluate(self.testX, self.testY, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def compile(self, loss, optimizer, metrics):
        print("compiling model_efficientnet...")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        print("done")

    def train(self, batch_size=128, epochs=15, **kwargs):
        use_callback = kwargs.get('no_callback')
        callback_path = kwargs.get('callback')
        if use_callback is None or use_callback:
            custom_callback = KerasCallback(model_to_save=self, use_callback=True, filepath=callback_path)
        else:
            custom_callback = KerasCallback(model_to_save=self, use_callback=False, filepath=callback_path)
        if isinstance(self.model, keras.Model):
            print("training model_efficientnet ...")
            print(f"batch-size = {batch_size}, epochs = {epochs}")
            print(f"train X len = {len(self.trainX)} test X len = {len(self.testX)}")
            self.history = self.model.fit(self.trainX, self.trainY,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          callbacks=[custom_callback],
                                          **kwargs)
            print("done")
        else:
            print("can`t start training. Model is not set")

    def predict(self, X: list, Y: list = None):
        transform_X = np.asarray([self.dataset.parser.file_data_transform(X_el) for X_el in X])
        transform_X = self.dataset.parser.input_transformer(transform_X)

        prediction = self.dataset.parser.prediction_transform(self.model.predict(transform_X))
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
