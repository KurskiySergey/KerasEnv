from keras_import import *


class TestLayer(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return tf.argmax(inputs, axis=-1)
