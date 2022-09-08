import os
from config import DATASET_FOLDER, MODELS_FOLDER, KERAS_FOLDER, ONNX_FOLDER, PREDICTION_FOLDER

if __name__ == "__main__":

    folders = [
        DATASET_FOLDER,
        MODELS_FOLDER,
        KERAS_FOLDER,
        ONNX_FOLDER,
        PREDICTION_FOLDER,
        "tests"
    ]

    for folder in folders:
        try:
            os.mkdir(folder)
        except OSError:
            print(f"{folder} already exists")

    open("tests/datatest.py", 'w').close()

