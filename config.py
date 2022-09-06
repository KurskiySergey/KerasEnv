import os

ONNX_FOLDER = "onnx_models"
KERAS_FOLDER = "keras_models"
PREDICTION_FOLDER = "predictions"
DATASET_FOLDER = "datasets"
MODELS_FOLDER = "models"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ONNX_DIR = os.path.join(BASE_DIR, ONNX_FOLDER)
KERAS_DIR = os.path.join(BASE_DIR, KERAS_FOLDER)
PREDICTIONS_DIR = os.path.join(BASE_DIR, PREDICTION_FOLDER)
DATASETS_DIR = os.path.join(BASE_DIR, DATASET_FOLDER)
MODELS_DIR = os.path.join(BASE_DIR, MODELS_FOLDER)
