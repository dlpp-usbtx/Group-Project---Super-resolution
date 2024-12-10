# Custom layer wrapping tf.nn.depth_to_space
import tensorflow as tf
from tensorflow.keras.layers import Layer


class DepthToSpaceLayer(Layer):
    def __init__(self, block_size, **kwargs):
        super(DepthToSpaceLayer, self).__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.block_size)

    def get_config(self):
        config = super(DepthToSpaceLayer, self).get_config()
        config.update({"block_size": self.block_size})
        return config