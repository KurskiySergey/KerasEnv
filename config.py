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

SETTINGS_FILE = "settings.ini"
SETTINGS_PATH = os.path.join(BASE_DIR, SETTINGS_FILE)

HDF5_FOLDER = "hdf5"
HDF5_DIR = os.path.join(BASE_DIR, HDF5_FOLDER)

SFTP_HDF5_FOLDER = "sftp_hdf5"
SFTP_HDF5_DIR = os.path.join(HDF5_DIR, SFTP_HDF5_FOLDER)

UNION_HDF5_FOLDER = "union_hdf5"
UNION_HDF5_DIR = os.path.join(HDF5_DIR, UNION_HDF5_FOLDER)

REAL_TIME_FOLDER = "real_time_tests"
REAL_TIME_DIR = os.path.join(BASE_DIR, "tests", REAL_TIME_FOLDER)

TRAIN_HDF5_FOLDER = "train_hdf5"
TRAIN_HDF5_DIR = os.path.join(HDF5_DIR, TRAIN_HDF5_FOLDER)

MASK_HDF5_FOLDER = "mask_hdf5"
MASK_HDF5_DIR = os.path.join(HDF5_DIR, MASK_HDF5_FOLDER)

MASK_INPUT_DIR = os.path.join(MASK_HDF5_DIR, 'input')
MASK_OUTPUT_DIR = os.path.join(MASK_HDF5_DIR, 'output')
MASK_RAW_DIR = os.path.join(MASK_HDF5_DIR, "raw_mask")

MODELS_IMG_FOLDER = "models_img"
MODELS_IMG_DIR = os.path.join(BASE_DIR, MODELS_IMG_FOLDER)

EVOLUTION_RESULTS_FOLDER = "evolution_results"
EVOLUTION_RESULTS_DIR = os.path.join(BASE_DIR, "tests", EVOLUTION_RESULTS_FOLDER)

SensorData_FOLDER = "sensor_data"
SensorData_Dir = os.path.join(BASE_DIR, SensorData_FOLDER)
