import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim 

    def build(self, input_shape):
        self.embedding = self.add_weight("embedding", shape=[input_shape[-1],self.emb_dim])
        self.bias = self.add_weight("bias", shape=(self.emb_dim,))


    def call(self, inputs):
        return inputs@self.embedding + self.bias