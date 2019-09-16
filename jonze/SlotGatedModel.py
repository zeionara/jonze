import tensorflow as tf

from . import CustomBahdanauAttention
from . import BahdanauAttention
from . import SlotGate
from . import CustomDropout

from tensorflow.compat.v1.keras.layers import CuDNNLSTM

class SlotGatedModel(tf.keras.Model):
    def __init__(self, input_size, slot_size, intent_size, layer_size=128):
        super(SlotGatedModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_size, layer_size)
        self.bilstm = tf.keras.layers.Bidirectional(CuDNNLSTM(layer_size, return_sequences=True, return_state=True))
        self.dropout = CustomDropout.CustomDropout(0.5)

        self.attn_size = 2 * layer_size
        self.slot_att = CustomBahdanauAttention.CustomBahdanauAttention(self.attn_size)
        self.intent_att = BahdanauAttention.BahdanauAttention(self.attn_size)
        self.slot_gate = SlotGate.SlotGate(self.attn_size)

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

        slot_d, _ = self.slot_att(state_outputs, state_outputs)
        intent_d, _ = self.intent_att(final_state, state_outputs)

        intent_fd = tf.keras.layers.concatenate([intent_d, final_state], -1)
        slot_gated, _ = self.slot_gate(intent_fd, slot_d)
        slot_fd = tf.keras.layers.concatenate([slot_gated, state_outputs], -1)

        intent = self.intent_out(intent_fd)
        slots = self.slot_out(slot_fd)
        outputs = [slots, intent]
        return outputs