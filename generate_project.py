from argparse import ArgumentParser
import os
from config import DATASETS_DIR, MODELS_DIR, BASE_DIR
from PIL import Image
from random import randint
PARSER = ArgumentParser("model generation")
PARSER.add_argument("--filename", "-f", type=str, help="name of the file", default="keras_env_model")
PARSER.add_argument("--dataset", "-d", type=str, help="name of te dataset", default="keras_env_dataset")
ARGS = PARSER.parse_args()


def generate_dataset():
    dataset_dir = os.path.join(DATASETS_DIR, ARGS.dataset)
    try:
        os.mkdir(dataset_dir)
        os.chdir(dataset_dir)

        os.mkdir("input")
        os.mkdir("output")
        os.mkdir("raw_test")

        for dir_name in ["test", "train"]:
            os.mkdir(dir_name)
            os.mkdir(os.path.join(dir_name, "input"))
            os.mkdir(os.path.join(dir_name, "output"))
    except OSError:
        print("dataset already exists")
    finally:
        os.chdir(BASE_DIR)
        generate_raw_test_simple_data(dataset_raw_dir=os.path.join(dataset_dir, "raw_test"))


def generate_base_import():
    data = "import keras\n" \
           "from keras_import import *\n" \
           "from config import DATASETS_DIR, KERAS_DIR\n" \
           "from keras_model import Model\n" \
           "from datasets import Dataset\n" \
           "from parsers import Parser\n" \
           "from PIL import Image, ImageOps\n\n"

    return data


def generate_base_model():

    data = f"class {ARGS.filename.title()}Model(Model):\n\n" \
           "\tdef create_model(self):\n" \
           "\t\tmodel = keras.Sequential([\n" \
           "\t\t\t\tkeras.Input(shape=(100, 100, 1)),\n" \
           "\t\t\t\tlayers.Flatten(),\n" \
           "\t\t\t\tlayers.Dense(50, activation='relu'),\n" \
           "\t\t\t\tlayers.Dense(3, activation='softmax')\n" \
           "\t\t])\n" \
           "\t\tself.model = model\n\n" \
           "\tdef save_prediction(self):\n" \
           "\t\tpass\n\n"

    return data


def generate_base_dataset():
    data = f"class {ARGS.filename.title()}Dataset(Dataset):\n" \
           "\tpass\n\n\n"

    return data


def generate_base_parser():

    data = f"class {ARGS.filename.title()}Parser(Parser):\n" \
           f"\tdef file_parse_func(self, filename: str, is_input=True) -> np.ndarray:\n\n" \
           f"\t\tif is_input:\n" \
           f"\t\t\timage = Image.open(filename)\n" \
           f"\t\t\tresult = self.file_data_transform(file=image)\n" \
           f"\t\t\tself.add_info['input_shape'] = result.shape\n" \
           f"\t\t\treturn result\n" \
           f"\t\telse:\n" \
           f"\t\t\treturn np.asarray(int(filename))\n\n" \
           f"\tdef file_data_transform(self, file):\n" \
           f"\t\tfile = ImageOps.grayscale(file)\n" \
           f"\t\tresult = np.expand_dims(np.asarray(file), -1)\n" \
           f"\t\treturn result\n\n" \
           f"\tdef input_transformer(self, data: np.ndarray) -> np.ndarray:\n" \
           f"\t\tmax_value = np.max(data)\n" \
           f"\t\treturn data / max_value\n\n" \
           f"\tdef output_transformer(self, data: np.ndarray) -> np.ndarray:\n" \
           f"\t\treturn np.argmax(data, axis=1)\n\n" \
           f"\tdef one_file_parse(self, filename: str) -> (np.ndarray, np.ndarray):\n" \
           f"\t\tinput_data = []\n" \
           f"\t\toutput_data = []\n" \
           f"\t\twith open(filename, 'r') as file:\n" \
           f"\t\t\tfor line in file:\n" \
           f"\t\t\t\timage_name, status = line.rstrip().split(" ")\n" \
           "\t\t\t\tinput_data.append(self.file_parse_func(image_name, is_input=True))\n" \
           "\t\t\t\toutput_data.append(self.file_parse_func(status, is_input=False))\n" \
           "\t\treturn np.asarray(input_data), np.asarray(output_data)\n\n" \
           "\tdef show_data(self, train: list, test: list, per_of_data: float):\n" \
           "\t\tpass\n\n" \
           "\tdef prediction_transform(self, X):\n" \
           "\t\tprint('Origin result')\n" \
           "\t\tprint(X)\n" \
           "\t\treturn np.argmax(X, axis=1)\n\n"

    return data


def generate_raw_test_simple_data(dataset_raw_dir):
    image = Image.new(mode="L", size=(100, 100))
    for i in range(100):
        for j in range(100):
            image.putpixel((i, j), value=randint(0, 255))

    image.save(f"{dataset_raw_dir}/test.png")


def generate_base_functions():

    prepare_data = f"def prepare_model():\n" \
                   f"\t{ARGS.filename.lower()}_dataset = {ARGS.filename.title()}" \
                   f"Dataset(dir_name=os.path.join(DATASETS_DIR, ARGS.dataset))\n" \
                   f"\t{ARGS.filename.lower()}_parser = {ARGS.filename.title()}Parser()\n" \
                   f"\t{ARGS.filename.lower()}_model = {ARGS.filename.title()}Model()\n\n" \
                   f"\t{ARGS.filename.lower()}_dataset.set_parser({ARGS.filename.lower()}_parser)\n" \
                   f"\t{ARGS.filename.lower()}_model.set_dataset({ARGS.filename.lower()}_dataset)\n\n" \
                   f"\tif not ARGS.predict:\n" \
                   f"\t\t{ARGS.filename.lower()}_model.load_dataset()\n" \
                   f"\t{ARGS.filename.lower()}_model.create_model()\n\n" \
                   f"\t{ARGS.filename.lower()}_loss='binary_crossentropy'\n" \
                   f"\t{ARGS.filename.lower()}_optimizer = 'adam'\n" \
                   f"\t{ARGS.filename.lower()}_metrics = ['accuracy']\n" \
                   f"\t{ARGS.filename.lower()}_model.compile(loss={ARGS.filename.lower()}_loss, " \
                   f"optimizer={ARGS.filename.lower()}_optimizer, " \
                   f"metrics={ARGS.filename.lower()}_metrics)\n\n" \
                   f"\treturn {ARGS.filename.lower()}_model\n\n"

    test_data = f"def test_model():\n" \
                f"\tmodel = prepare_model()\n" \
                f"\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.model))\n" \
                f"\tmodel.test()\n\n"

    train_data = f"def train_model():\n" \
                 f"\tmodel = prepare_model()\n" \
                 f"\tif ARGS.train_model:\n" \
                 f"\t\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.model))\n\n" \
                 f"\tbatch_size = ARGS.batch_size\n" \
                 f"\tepochs = ARGS.epochs\n\n" \
                 f"\tmodel.train(batch_size=batch_size, epochs=epochs)\n" \
                 f"\tmodel.test()\n\n" \
                 f"\tif ARGS.save_model:\n" \
                 f"\t\tmodel.save_model(os.path.join(KERAS_DIR, ARGS.filename))\n\n"

    predict_data = f"def prediction_test():\n" \
                   f"\tmodel = prepare_model()\n" \
                   f"\tif not ARGS.raw_model:\n" \
                   f"\t\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.filename))\n\n" \
                   f"\tpred_data, real_data = get_predict_data()\n" \
                   f"\tmodel.predict(pred_data, Y=real_data)\n\n" \
                   f"\tif ARGS.save_predict:\n" \
                   f"\t\tmodel.save_prediction()\n\n"

    get_data = "def get_predict_data():\n" \
               "\traw_info_dir = os.path.join(DATASETS_DIR, ARGS.dataset, 'raw_test')\n" \
               "\tpred_data = [Image.open(os.path.join(raw_info_dir, filename)) " \
               "for filename in os.listdir(raw_info_dir)]\n" \
               "\treal_data = ['Just test']\n" \
               "\treturn pred_data, real_data\n\n"

    prepare_dataset = "def prepare_dataset():\n" \
                      "\tpass\n\n"

    return prepare_data+test_data+train_data+predict_data+get_data+prepare_dataset


def generate_main_function():

    main_data = "def main():\n\n" \
                "\tif ARGS.predict:\n" \
                "\t\tprediction_test()\n" \
                "\telse:\n" \
                "\t\tif ARGS.test:\n" \
                "\t\t\ttest_model()\n" \
                "\t\telse:\n" \
                "\t\t\ttrain_model()\n\n"

    return main_data


def generate_start_data():
    start_data = "if __name__ == '__main__':\n" \
                 "\tmain()\n\n"

    return start_data


def generate_file():
    base_data = [
        generate_base_import(),
        generate_base_dataset(),
        generate_base_parser(),
        generate_base_model(),
        generate_base_functions(),
        generate_main_function(),
        generate_start_data()
    ]

    file_data = "".join(base_data)

    filename = os.path.join(MODELS_DIR, ARGS.filename)
    with open(f"{filename}.py", "w") as file:
        file.write(file_data)


def main():
    generate_dataset()
    generate_file()


if __name__ == "__main__":
    main()
