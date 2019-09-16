import tensorflow as tf
from tensorflow.keras import layers


class CustomDropout(layers.Layer):
  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
        return tf.nn.dropout(inputs, rate=self.rate)
    return inputs