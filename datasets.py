import numpy as np
import os
import random
from parsers import Parser, TestParser, MnistParser, SDTestParser


class Dataset:
    def __init__(self, filename: str = None, dir_name: str = None, per_for_test: int = 13, use_dataset=True, shuffle_data=False, batch_size=None):
        self.test_data: list = [[], []]
        self.train_data: list = [[], []]
        self.filename = filename
        self.dir_name = dir_name
        self.per_for_test = per_for_test
        self.parser: Parser = None
        self.use_dataset = use_dataset
        self.shuffle_data = shuffle_data
        self.batch_size = batch_size
        self.train_samples = 0
        self.test_samples = 0

    def load_dataset(self, fit_generator=False, batch_size=10):
        if self.use_dataset:
            if not fit_generator:
                if self.shuffle_data:
                    self.__load_shuffled_data()
                else:
                    self.__load_origin_data()
                self.__get_samples()
            else:
                self.__load_generator()


    def show_data(self, per_of_data: float = 10):
        print(f"train input/output length: {len(self.train_data[0])}")
        print(f"test input/output length: {len(self.test_data[0])}")
        print(f"train/test input shape: {self.train_data[0].shape}")
        print(f"train/test output shape: {self.test_data[1].shape}")

        self.parser.show_data(self.train_data, self.test_data, per_of_data=per_of_data)

    def set_parser(self, parser: Parser):
        self.parser = parser

    def __load_generator(self):
        train_dir = os.path.join(self.dir_name, "train")
        test_dir = os.path.join(self.dir_name, "test")
        self.train_data = [self.raw_generator(files_dir=train_dir, use_batches=True, one_use=False), None]
        self.test_data = [self.raw_generator(files_dir=test_dir, use_batches=True, one_use=True), None]

    def __get_samples(self):
        if self.batch_size is not None:
            self.test_samples = int(len(self.test_data[1]) // self.batch_size)
            self.train_samples = int(len(self.train_data[1]) // self.batch_size)

            if self.train_samples * self.batch_size != len(self.train_data[1]):
                self.train_samples += 1
            if self.test_samples * self.batch_size != len(self.test_data[1]):
                self.test_samples += 1

    def __load_shuffled_data(self):
        if self.filename is None:
            if self.dir_name is None:
                print("No dataset")
            else:
                input_data_path = os.path.join(self.dir_name, "input")
                output_data_path = os.path.join(self.dir_name, "output")
                input_data = self.__load_from_dir_name(input_data_path, is_input=True)
                output_data = self.__load_from_dir_name(output_data_path, is_input=False)

                input_data = self.parser.input_transformer(input_data)
                output_data = self.parser.output_transformer(output_data)
                self.test_data, self.train_data = self.__shuffle_data(input_data, output_data)
        else:
            input_data, output_data = self.__load_from_filename(self.filename, one_file=True)
            input_data = self.parser.input_transformer(input_data)
            output_data = self.parser.output_transformer(output_data)
            self.test_data, self.train_data = self.__shuffle_data(input_data, output_data)

    def __load_origin_data(self):
        if self.filename is None:
            if self.dir_name is None:
                print("No dataset")
            else:
                test_input_path = os.path.join(self.dir_name, "test", "input")
                test_output_path = os.path.join(self.dir_name, "test", "output")
                train_input_path = os.path.join(self.dir_name, "train", "input")
                train_output_path = os.path.join(self.dir_name, "train", "output")

                test_input_data = self.__load_from_dir_name(test_input_path, is_input=True)
                test_output_data = self.__load_from_dir_name(test_output_path, is_input=False)
                self.test_data = [self.parser.input_transformer(test_input_data),
                                  self.parser.output_transformer(test_output_data)]

                train_input_data = self.__load_from_dir_name(train_input_path, is_input=True)
                train_output_data = self.__load_from_dir_name(train_output_path, is_input=False)
                self.train_data = [self.parser.input_transformer(train_input_data),
                                   self.parser.output_transformer(train_output_data)]
        else:
            test_input_data, train_input_data, test_output_data, train_output_data = \
                self.__load_from_filename(self.filename, one_file=True)

            test_input_data = self.parser.input_transformer(test_input_data)
            train_input_data = self.parser.input_transformer(train_input_data)
            test_output_data = self.parser.output_transformer(test_output_data)
            train_output_data = self.parser.output_transformer(train_output_data)

            self.train_data = [train_input_data, train_output_data]
            self.test_data = [test_input_data, test_output_data]

    def __load_from_dir_name(self, dir_name, is_input=True) -> np.ndarray:
        filenames = os.listdir(dir_name)
        try:
            filenames = sorted(filenames, key=lambda filename: int(filename.split(".")[0]))
            # print(filenames)
        except ValueError:
            pass
            # print(filenames)
        result = []
        # print(filenames)
        for file in filenames:
            file = os.path.join(dir_name, file)
            data = self.__load_from_filename(file, is_input, one_file=False)
            result.append(data)

        result = np.asarray(result)
        print(f"is input = {is_input} ", result.shape)
        return result

    def __load_from_filename(self, filename, is_input=True, one_file=False):
        if one_file:
            data = self.parser.one_file_parse(filename)
        else:
            data = self.parser.file_parse_func(filename=filename, is_input=is_input)
        return data

    def __shuffle_data(self, in_data: np.ndarray, out_data: np.ndarray):
        sample_numbers = len(in_data)
        test_sample_number = int(sample_numbers / 100 * self.per_for_test)
        used_samples = []
        test_in = []
        test_out = []

        sample_number = random.randint(0, sample_numbers-1)
        for i in range(test_sample_number):
            while sample_number in used_samples:
                sample_number = random.randint(0, sample_numbers-1)
            used_samples.append(sample_number)
            test_in.append(in_data[sample_number])
            test_out.append(out_data[sample_number])

        test_in = np.asarray(test_in)
        test_out = np.asarray(test_out)

        train_in = np.delete(in_data, used_samples, axis=0)
        train_out = np.delete(out_data, used_samples, axis=0)

        print("train shape")
        print("in ", train_in.shape, "out ", train_out.shape)

        print("test shape")
        print("in ", test_in.shape, "out ", test_out.shape)
        # print(train_in)
        # print(train_out)

        return [test_in, test_out], [train_in, train_out]

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_val_data(self):
        return self.test_data

    def generator_transform(self, data):
        return {data[0]}, {data[1]}

    def raw_generator(self, files_dir, is_input=True, one_use=False, use_batches=False, in_out_split = True):
        if not in_out_split:
            files = [os.path.join(files_dir, file) for file in sorted(os.listdir(files_dir))]
            while True:
                if use_batches:
                    for index in range(len(files)):
                        files_dt = files[index*self.batch_size:(index+1)*self.batch_size]
                        in_data = []
                        out_data = []
                        for file in files_dt:
                            input_dt, output_dt = self.parser.file_parse_func(file, is_input=is_input)
                            in_data.append(input_dt)
                            out_data.append(output_dt)
                        yield self.generator_transform([np.asarray(in_data), np.asarray(out_data)])
                else:
                    for file in files:
                        result = self.parser.file_parse_func(file, is_input=is_input)
                        yield self.generator_transform(result)
                if one_use:
                    break
        else:
            in_dir = os.path.join(files_dir, "input")
            out_dir = os.path.join(files_dir, "output")
            in_files = [os.path.join(in_dir, file) for file in sorted(os.listdir(in_dir))]
            out_files = [os.path.join(out_dir, file) for file in sorted(os.listdir(out_dir))]
            while True:
                if use_batches:
                    for index in range(len(in_files)):
                        in_files_dt = in_files[index*self.batch_size:(index+1)*self.batch_size]
                        out_files_dt = out_files[index*self.batch_size:(index+1)*self.batch_size]
                        in_data = []
                        out_data = []
                        for in_file, out_file in zip(in_files_dt, out_files_dt):
                            input_dt = self.parser.file_parse_func(in_file, is_input=True)
                            out_dt = self.parser.file_parse_func(out_file, is_input=False)
                            in_data.append(input_dt)
                            out_data.append(out_dt)
                        yield self.generator_transform([np.asarray(in_data), np.asarray(out_data)])
                else:
                    for in_file, out_file in zip(in_files, out_files):
                        in_dt = self.parser.file_parse_func(in_file, is_input=True)
                        out_dt = self.parser.file_parse_func(out_file, is_input=False)
                        yield self.generator_transform([in_dt, out_dt])
                if one_use:
                    break


class TestDataset(Dataset):
    pass


class MnistDataset(Dataset):

    def load_dataset(self, shuffle_data=True):
        from keras.datasets import mnist
        (trainX, trainY), (testX, testY) = mnist.load_data()
        self.test_data = [self.parser.input_transformer(testX), self.parser.output_transformer(testY)]
        self.train_data = [self.parser.input_transformer(trainX), self.parser.output_transformer(testY)]


def test_dataset():
    dataset = TestDataset(dir_name="test/train", shuffle_data=True)
    test_parser = SDTestParser()
    dataset.set_parser(test_parser)
    dataset.load_dataset()
    dataset.show_data(per_of_data=1)


def mnist_test_dataset():
    mnist_dataset = MnistDataset()
    test_parser = MnistParser()
    mnist_dataset.set_parser(test_parser)
    mnist_dataset.load_dataset()


if __name__ == "__main__":
    test_dataset()
    # mnist_test_dataset()
