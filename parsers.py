import random
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot


class Parser(ABC):

    def __init__(self):
        super().__init__()
        self.add_info = {}

    @abstractmethod
    def file_parse_func(self, filename: str, is_input=True) -> np.ndarray:
        pass

    @abstractmethod
    def file_data_transform(self, file):
        pass

    @abstractmethod
    def label_transform(self, label):
        pass

    @abstractmethod
    def input_transformer(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def output_transformer(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def one_file_parse(self, filename: str) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def show_data(self, train: list, test: list, per_of_data: float):
        pass

    @abstractmethod
    def prediction_transform(self, X):
        pass


class TestParser(Parser):

    def file_data_transform(self, file):
        file = ImageOps.grayscale(file)
        result = np.expand_dims(np.asarray(file), -1)
        return result

    def file_parse_func(self, filename: str, is_input=True) -> np.ndarray:
        if is_input:
            image = Image.open(filename)
            result = self.file_data_transform(file=image)
            # result = np.asarray(image)
            self.add_info["input_shape"] = result.shape
            return result
        else:
            with open(filename, "r") as f:
                data = [int(i) for i in f.readline().split(" ")]
                return np.asarray(data)

    def input_transformer(self, data: np.ndarray) -> np.ndarray:
        data = data.astype("float32") / 255
        result = []
        for sample in data:
            result.append(sample.flatten(order="C"))
        data = np.asarray(result)
        # print(data)
        # print(data.shape)
        return data

    def output_transformer(self, data: np.ndarray) -> np.ndarray:
        data = data[0]
        number_of_classes = np.max(data, axis=0) + 1
        result = []
        for index in data:
            vector = []
            for i in range(number_of_classes):
                vector.append(1) if i == index else vector.append(0)
            result.append(vector)
        result = np.asarray(result)
        # print(result)
        # print(result.shape)
        return result

    def prediction_transform(self, X):
        return X

    def one_file_parse(self, filename: str) -> np.ndarray:
        pass

    def show_data(self, train: list, test: list, per_of_data: float = 10):
        train_samples_number = int(len(train[0]) * per_of_data/100)
        test_samples_number = int(len(test[0]) * per_of_data/100)

        train_samples_index = [random.randint(0, len(train[0])) for _ in range(train_samples_number)]
        test_samples_index = [random.randint(0, len(test[0])) for _ in range(test_samples_number)]

        train_y = np.asarray([np.where(train_y_sample == 1)[0][0]
                              for i, train_y_sample in enumerate(train[1]) if i in train_samples_index])
        train_x = np.asarray([train_x_sample.reshape(self.add_info.get("input_shape")[:-1])
                              for i, train_x_sample in enumerate(train[0])
                              if i in train_samples_index]).astype("float32") * 255

        test_y = np.asarray([np.where(test_y_sample == 1)[0][0]
                            for i, test_y_sample in enumerate(test[1]) if i in test_samples_index])
        test_x = np.asarray([test_x_sample.reshape(self.add_info.get("input_shape")[:-1])
                            for i, test_x_sample in enumerate(test[0])
                             if i in test_samples_index]).astype("float32") * 255

        subplot_sq = 3
        plots_number = train_samples_number // subplot_sq + 1
        train_len = len(train_x)
        train_i = 0
        for _ in range(plots_number):
            for i in range(subplot_sq*subplot_sq):
                if train_i == train_len:
                    break
                else:
                    print(train_x[train_i].shape)
                    pyplot.subplot(int(f"{subplot_sq}{subplot_sq}{i+1}"))
                    pyplot.title(f"{train_y[train_i]}")
                    pyplot.imshow(train_x[train_i], cmap=pyplot.get_cmap("gray"), vmin=0, vmax=255)
                    train_i += 1
            pyplot.show()


class TestParserConv(TestParser):

    def input_transformer(self, data: np.ndarray) -> np.ndarray:
        data = data.astype("float32") / 255
        return data


class MnistParser(Parser):

    def show_data(self, train: list, test: list, samples_number: int):
        pass

    def file_data_transform(self, file):
        pass

    def file_parse_func(self, filename: str, is_input=True) -> np.ndarray:
        pass

    def one_file_parse(self, filename: str) -> np.ndarray:
        pass

    def input_transformer(self, data: np.ndarray) -> np.ndarray:
        data = data.astype("float32") / 255
        data = np.expand_dims(data, -1)
        return data

    def output_transformer(self, data: np.ndarray) -> np.ndarray:
        number_of_classes = np.max(data, axis=0) + 1
        result = []
        for index in data:
            vector = []
            for i in range(number_of_classes):
                vector.append(1) if i == index else vector.append(0)
            result.append(vector)
        result = np.asarray(result)
        return result

    def prediction_transform(self, X):
        return X


class SDTestParser(Parser):

    def file_data_transform(self, file):
        image = Image.open(file)
        image = ImageOps.grayscale(image)
        result = np.asarray(image)
        return result

    def show_data(self, train: list, test: list, per_of_data: float):
        first = train[0][17]
        first_out = train[1][17]
        print(first.shape)
        print(first_out.shape)

        def get_value(value):

            result = np.where(value == 1)
            if result[0][0] == 1:
                return 1
            else:
                return 0

        v_func = np.vectorize(get_value, signature="(n)->()")
        result = np.array(v_func(first_out).tolist())
        first = first.astype("float32") * 255
        result = result.astype("float32") * 255
        img_input = Image.fromarray(first.astype(np.uint8))
        img_output = Image.fromarray(result.astype(np.uint8))
        img_input.show()
        img_output.show()
        pass

    def input_transformer(self, data: np.ndarray) -> np.ndarray:
        data = np.expand_dims(data, -1)
        # print(len()
        # input()
        data = data.astype("float32") / 255

        def update_value(value):
            if value < 0.7:
                return value
            else:
                return 1

        v_func = np.vectorize(update_value)
        data = np.array(v_func(data).tolist())

        return data

    def output_transformer(self, data: np.ndarray) -> np.ndarray:
        # print(np.unique(data[0]))
        data = data.astype("float32") / 255

        num_classes = 2

        # print(data[0])
        # print(np.unique(data[0]))
        def get_vector(value, num_classes):
            if value < 0.7:
                return np.asarray([1, 0])
            else:
                return np.asarray([0, 1])

        v_func = np.vectorize(get_vector, otypes=[np.ndarray])
        data = np.array(v_func(data, num_classes).tolist())
        # print(data[0])
        return data

    def prediction_transform(self, X):
        result = []
        for k in range(len(X)):
            X_0_raw = [[data[0] for data in X[0][i]] for i in range(len(X[k]))]
            X_1_raw = [[data[1] for data in X[0][i]] for i in range(len(X[k]))]
            result_0 = np.argmax(X_0_raw, axis=1)
            result_0_total = np.argmax(X_0_raw)
            # print(result_0)
            # print(result_0_total, X[k].shape)
            max_0_i = result_0_total // X[k].shape[0]
            max_0_j = result_0_total - max_0_i * X[k].shape[0]
            # print(max_0_i, max_0_j, X_0_raw[max_0_i][max_0_j])

            result_1 = np.argmax(X_1_raw, axis=1)
            result_1_total = np.argmax(X_1_raw)
            max_1_i = result_1_total // X[k].shape[0]
            max_1_j = result_1_total - max_1_i * X[k].shape[0]
            # print(result_1)

            X_0 = []
            X_1 = []

            for i in range(len(X_0_raw)):
                line = []
                for j in range(len(X_0_raw[i])):
                    # if j == result_0[i]:
                    #     line.append(1)
                    # else:
                    #     # print(X_0_raw[i][result_0[i]])
                    #     line.append(X_0_raw[i][j]/X_0_raw[i][result_0[i]])
                    #     # line.append(1)
                    line.append(X_0_raw[i][j]/X_0_raw[max_0_i][max_0_j])
                X_0.append(line)

            for i in range(len(X_1_raw)):
                line = []
                for j in range(len(X_1_raw[i])):
                    # if j == result_1[i]:
                    #     line.append(1)
                    # else:
                    #     # print(X_1_raw[i][result_1[i]])
                    #     line.append(X_1_raw[i][j]/X_1_raw[i][result_1[i]])
                    #     # line.append(1)
                    line.append(X_1_raw[i][j]/X_1_raw[max_1_i][max_1_j])
                X_1.append(line)

            X_0 = np.asarray(X_0).astype("float32") * 255
            X_1 = np.asarray(X_1).astype("float32") * 255

            img_0 = Image.fromarray(X_0).convert("RGB")
            img_1 = Image.fromarray(X_1).convert("RGB")

            # img_0.show()
            img_1.show()

            result.append([img_0, img_1])
        return result

    def one_file_parse(self, filename: str) -> np.ndarray:
        pass

    def file_parse_func(self, filename: str, is_input=True) -> np.ndarray:
        data = self.file_data_transform(filename)
        if is_input:
            self.add_info["input_shape"] = data.shape
        else:
            self.add_info["output_shape"] = data.shape

        return data
