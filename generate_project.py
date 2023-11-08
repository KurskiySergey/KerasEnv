from argparse import ArgumentParser
import os
from config import DATASETS_DIR, MODELS_DIR, BASE_DIR
from PIL import Image
from random import randint
PARSER = ArgumentParser("model generation")
PARSER.add_argument("--filename", "-f", type=str, help="name of the file", default="keras_env_model")
PARSER.add_argument("--dataset", "-d", type=str, help="name of te dataset", default="keras_env_dataset")
PARSER.add_argument("--folder", action="store_true", default=False)
ARGS = PARSER.parse_args()


def generate_dataset():
    dataset_dir = os.path.join(DATASETS_DIR, ARGS.dataset)
    try:
        os.mkdir(dataset_dir)
        os.chdir(dataset_dir)

        os.mkdir("input")
        os.mkdir("output")
        os.mkdir("raw_test")
        os.mkdir("history")

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
           "from callbacks import KerasCallback\n" \
           "from config import DATASETS_DIR, KERAS_DIR\n" \
           "from keras_model import Model\n" \
           "from datasets import Dataset\n" \
           "from parsers import Parser\n" \
           "from PIL import Image, ImageOps\n\n"

    return data


def generate_base_model_import():
    data = "import keras\n" \
           "from keras_import import *\n" \
           "from keras_model import Model\n\n" \

    return data

def generate_base_dataset_import():
    data = "from datasets import Dataset\n\n"

    return data

def generate_base_parser_import():
    data = "from keras_import import *\n" \
           "from parsers import Parser\n" \
           "from PIL import Image, ImageOps\n\n"

    return data

def generate_base_folder_functions_import():
    filename = ARGS.filename
    data = ("import keras\n"
            "from keras_import import *\n"
            "from callbacks import KerasCallback\n"
            "from config import DATASETS_DIR, KERAS_DIR\n"
            f"from models.{filename.lower()}.{filename.title()}Dataset import {filename.title()}Dataset\n"
            f"from models.{filename.lower()}.{filename.title()}Parser import {filename.title()}Parser\n"
            f"from models.{filename.lower()}.{filename.title()}Model import {filename.title()}Model\n\n")

    return "".join(data)

def generate_base_run_import():

    filename = ARGS.filename
    data = (
             "from keras_import import *\n"
             "from config import DATASETS_DIR\n"
             f"from models.{filename.lower()}.{filename.title()}BaseFunctions import prediction_test, test_model, train_model, prepare_dataset, show_model, test_dataset\n\n")

    return "".join(data)


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
           f"\t\t\tresult = self.file_data_transform(filename)\n" \
           f"\t\t\tself.add_info['input_shape'] = result.shape\n" \
           f"\t\t\treturn result\n" \
           f"\t\telse:\n" \
           f"\t\t\treturn self.label_transform(filename)\n\n" \
           f"\tdef file_data_transform(self, file):\n"\
           f"\t\tfile = Image.open(file)\n"\
           f"\t\tfile = ImageOps.grayscale(file)\n" \
           f"\t\tresult = np.expand_dims(np.asarray(file), -1)\n" \
           f"\t\treturn result\n\n"\
           f"\tdef label_transform(self, label):\n"\
           f"\t\tone_hot_label = np.asarray([0 if i != int(label) else 1 for i in range(3)])\n"\
           f"\t\treturn one_hot_label\n\n"\
           f"\tdef input_transformer(self, data: np.ndarray) -> np.ndarray:\n" \
           f"\t\tmax_value = np.max(data)\n" \
           f"\t\treturn data / max_value\n\n" \
           f"\tdef output_transformer(self, data: np.ndarray) -> np.ndarray:\n" \
           f"\t\treturn data\n\n" \
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
                   f"\tcustom_callback = KerasCallback(model_to_save={ARGS.filename.lower()}_model, filepath=ARGS.callback)\n" \
                   f"\tcallbacks = [custom_callback]\n" \
                   f"\t{ARGS.filename.lower()}_model.callbacks = callbacks\n\n" \
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
                f"\tif not ARGS.raw_model:\n" \
                f"\t\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.model))\n" \
                f"\tmodel.test()\n" \
                f"\treturn model\n\n"

    train_data = f"def train_model():\n" \
                 f"\tmodel = prepare_model()\n" \
                 f"\tif not ARGS.raw_model:\n" \
                 f"\t\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.model))\n\n" \
                 f"\tbatch_size = ARGS.batch_size\n" \
                 f"\tepochs = ARGS.epochs\n\n" \
                 f"\tmodel.train(batch_size=batch_size, epochs=epochs, no_callback=ARGS.no_callback)\n" \
                 f"\tmodel.test()\n\n" \
                 f"\tif ARGS.save_model:\n" \
                 f"\t\tmodel.save_model(os.path.join(KERAS_DIR, ARGS.filename))\n" \
                 f"\treturn model\n\n"

    predict_data = f"def prediction_test():\n" \
                   f"\tmodel = prepare_model()\n" \
                   f"\tif not ARGS.raw_model:\n" \
                   f"\t\tmodel.load_model(model_name=os.path.join(KERAS_DIR, ARGS.model))\n\n" \
                   f"\tpred_data, real_data = get_predict_data()\n" \
                   f"\tmodel.predict(pred_data, Y=real_data)\n\n" \
                   f"\tif ARGS.save_predict:\n" \
                   f"\t\tmodel.save_prediction()\n\n"

    get_data = "def get_predict_data():\n" \
               "\traw_info_dir = os.path.join(DATASETS_DIR, ARGS.dataset, 'raw_test')\n" \
               "\tpred_data = [os.path.join(raw_info_dir, filename) " \
               "for filename in os.listdir(raw_info_dir)]\n" \
               "\treal_data = ['Just test']\n" \
               "\treturn pred_data, real_data\n\n"

    prepare_dataset = "def prepare_dataset():\n" \
                      "\tfrom PIL import Image\n" \
                      "\tfrom random import randint\n" \
                      "\twith open(os.path.join(DATASETS_DIR, ARGS.dataset, 'info.txt'), 'w') as info:\n" \
                      "\t\tfor i in range(100):\n" \
                      "\t\t\timage = Image.new(mode='L', size=(100, 100))\n" \
                      "\t\t\tfor j in range(100):\n" \
                      "\t\t\t\tfor k in range(100):\n" \
                      "\t\t\t\t\timage.putpixel((j, k), value=randint(0, 255))\n" \
                      "\t\t\timage_path = os.path.join(DATASETS_DIR, ARGS.dataset, 'input', f'{i}.png')\n" \
                      "\t\t\timage.save(image_path)\n" \
                      "\t\t\tinfo.write(f'{image_path} {randint(0, 3)}\\n')\n\n"

    test_dataset = "def test_dataset():\n" \
                   "\tdataset_dir = os.path.join(DATASETS_DIR, ARGS.dataset)\n" \
                   "\tinfo_name = 'info.txt'\n" \
                   f"\tdataset = {ARGS.filename.title()}Dataset(dir_name=dataset_dir, filename=os.path.join(dataset_dir, info_name), shuffle_data=True, per_for_test=15, batch_size=ARGS.batch_size)\n" \
                   f"\tparser = {ARGS.filename.title()}Parser()\n" \
                   "\tdataset.set_parser(parser)\n" \
                   "\tdataset.load_dataset()\n" \
                   "\tdataset.show_data()\n\n"

    show_model = 'def show_model():\n' \
                 '\tmodel = prepare_model()\n' \
                 '\tprint(model.model.summary())\n\n'

    return prepare_data+test_data+train_data+predict_data+get_data+prepare_dataset+test_dataset+show_model


def generate_main_function():

    main_data = "def main():\n\n" \
                "\tif ARGS.predict:\n" \
                "\t\tprediction_test()\n" \
                "\telse:\n" \
                "\t\tif ARGS.test:\n" \
                "\t\t\tmodel=test_model()\n" \
                "\t\telse:\n" \
                "\t\t\tmodel=train_model()\n" \
                "\n" \
                "\t\tif ARGS.save_history:\n" \
                "\t\t\tif ARGS.save_model:\n" \
                "\t\t\t\thistory_name = ARGS.filename\n" \
                "\t\t\telif ARGS.predict:\n" \
                "\t\t\t\thistory_name = ARGS.model\n" \
                "\t\t\telse:" \
                "\t\t\t\thistory_name = ARGS.history\n" \
                "\t\t\tmodel.save_history(file_name=os.path.join(DATASETS_DIR, ARGS.dataset, " \
                "'history', history_name))\n\n"

    return main_data


def generate_start_data():
    start_data = ("if __name__ == '__main__':\n"
                  "\tif ARGS.no_cuda:\n"
                  "\t\tuse_cpu()\n\n"
                  "\tif ARGS.prepare_dataset:\n"
                  "\t\tprepare_dataset()\n\n"
                  "\tif ARGS.show_model:\n"
                  "\t\tshow_model()\n\n"
                  "\tif ARGS.test_dataset:\n"
                  "\t\ttest_dataset()\n\n"
                  "main()\n\n")

    return "".join(start_data)


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

    filename = os.path.join(MODELS_DIR, ARGS.filename.lower())
    with open(f"{filename}.py", "w") as file:
        file.write(file_data)


def generate_file_folder():

    filename = ARGS.filename.lower()
    file_folder = os.path.join(MODELS_DIR, ARGS.filename.lower())

    try:
        os.mkdir(file_folder)
        os.chdir(file_folder)

        with open(f"{filename.title()}Model.py", "w") as model:
            info = [generate_base_model_import(),
                    generate_base_model()]
            model.write("".join(info))

        with open(f"{filename.title()}Parser.py", "w") as parser:
            info = [generate_base_parser_import(),
                    generate_base_parser()]
            parser.write("".join(info))

        with open(f"{filename.title()}Dataset.py", "w") as dataset:
            info = [generate_base_dataset_import(),
                    generate_base_dataset()]
            dataset.write("".join(info))

        with open(f"{filename.title()}BaseFunctions.py", "w") as base_func:
            info = [generate_base_folder_functions_import(),
                    generate_base_functions()]

            base_func.write("".join(info))

        with open(f"run.py", "w") as run:
            info = [generate_base_run_import(),
                    generate_main_function(),
                    generate_start_data()]
            run.write("".join(info))

    except OSError:
        print("model already exists")
    finally:
        os.chdir(BASE_DIR)


def main():
    generate_dataset()
    if ARGS.folder:
        generate_file_folder()
    else:
        generate_file()


if __name__ == "__main__":
    main()
