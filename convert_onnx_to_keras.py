import argparse
from config import *
import onnx
from onnx2keras import onnx_to_keras

# do on cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def convert(model_name, keras_model_name):
    model_name = os.path.join(ONNX_DIR, model_name)
    onnx_model = onnx.load_model(model_name)
    input_names = [inp.name for inp in onnx_model.graph.input]
    print(input_names)
    model = onnx_to_keras(onnx_model, input_names=input_names, change_ordering=True)
    model.summary()
    model.save(os.path.join(KERAS_DIR, keras_model_name))


def main():
    print("reading args...")
    parser = argparse.ArgumentParser(description="converter parser")
    parser.add_argument("--onnx-name", "-on", type=str, default=None)
    parser.add_argument("--keras-name", "-kn", type=str, default=None)

    args = parser.parse_args()
    onnx_name = args.onnx_name
    keras_name = args.keras_name
    print("done")

    print("start converting...")
    if onnx_name is not None:
        if keras_name is not None:
            convert(onnx_name, keras_name)
        else:
            keras_name = onnx_name.split('.')[0]
            convert(onnx_name, keras_name)
        print("done")
    else:
        print("Error. No onnx input name")


if __name__ == "__main__":
    main()

