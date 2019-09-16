import tensorflow as tf


class CustomBahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(CustomBahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units,kernel_regularizer = tf.keras.regularizers.l2(0.02))
        self.W2 = tf.keras.layers.Conv1D(units,5,1,'same')
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        #hidden_with_time_axis = tf.expand_dims(query, 1)
        hidden_with_time_axis = query

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        #context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights