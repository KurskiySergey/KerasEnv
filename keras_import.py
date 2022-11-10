import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets

PARSER = argparse.ArgumentParser(description="Keras mnist parser")
PARSER.add_argument("--no-cuda", action="store_true", default=False)
PARSER.add_argument("--save-model", action="store_true", default=False)
PARSER.add_argument("--epochs", type=int, default=15)
PARSER.add_argument("--test", action="store_true", default=False)
PARSER.add_argument("--filename", "-f", type=str, default="keras_mnist_model")
PARSER.add_argument("--predict", "-p", action="store_true", default=False)
PARSER.add_argument("--raw-model", action="store_true", default=False)
PARSER.add_argument("--batch-size", type=int, default=10)
PARSER.add_argument("--save-predict", action="store_true", default=False)
PARSER.add_argument("--train-model", action="store_true", default=False)
PARSER.add_argument("--model", "-m", type=str, default="sd_test")
PARSER.add_argument("--dataset", "-d", type=str, default="train")
PARSER.add_argument("--no-callback", action='store_true', default=False)
PARSER.add_argument('--callback', type=str, default='callback_model')
ARGS = PARSER.parse_args()


def use_cpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
