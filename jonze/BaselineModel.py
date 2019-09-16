import tensorflow as tf

from . import CustomDropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class BaselineModel(tf.keras.Model):
    def __init__(self, input_size, slot_size, intent_size, layer_size=128):
        super(BaselineModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_size, layer_size)
        self.bilstm = tf.keras.layers.Bidirectional(CuDNNLSTM(layer_size, return_sequences=True, return_state=True))
        self.dropout = CustomDropout.CustomDropout(0.5)
        self.intent_out = tf.keras.layers.Dense(intent_size, activation=None)
        self.slot_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(slot_size, activation=None))

    @tf.function
    def call(self, inputs, sequence_length, isTraining=True):
        x = self.embedding(inputs)
        state_outputs, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)

        state_outputs = self.dropout(state_outputs, isTraining)
        forward_h = self.dropout(forward_h, isTraining)
        backward_h = self.dropout(backward_h, isTraining)

        final_state = tf.keras.layers.concatenate([forward_h, backward_h])
        intent = self.intent_out(final_state)
        slots = self.slot_out(state_outputs)
        outputs = [slots, intent]
        return outputs