import numpy as np
import tensorflow as tf

def SoftMax(input):
    out = tf.exp(input)
    out = out/tf.reduce_sum(out, axis=1)
    return out

class LinearMap(tf.keras.layers.Layer):
    '''
    Simple linear layer for a linear mapping of inputs
    '''
    def __init__(self, inner_dim):
        self.inner_dim = inner_dim
    def build(self, input):
        self.map = self.add_weight("map", shape=(input[-1],self.inner_dim),
                                   initializer='random_normal', trainable=True)
        self.bias = self.add_weight("bias", shape=(self.inner_dim,),
                                    initializer='random_normal', trainable=True)
    def call(self, input):
        return input@self.map + self.bias

class AttentionHead(tf.keras.layers.Layer):
    '''
    Takes already embedded 'input = (query, key, value)'
    Initialization denotes whether causal mask needed or not
    '''
    def __init__(self, out_dim):
        self.out_dim = out_dim 
    
    def build(self, key):
        # Initialize Linear Maps
        self.key_dim = key.shape[-1]
        self.query_map = LinearMap(self.key_dim)
        self.key_map= LinearMap(self.key_dim)
        self.value_map = LinearMap(self.key_dim)


    def call(self, query, key, value, causal_mask = False):
        Q = self.query_map(query)
        K = self.key_map(key)
        soft_weights = tf.matmul(Q,tf.transpose(K))
        if causal_mask:
            # Option 1: Take only the lower triangular portion: dependant on known keys
            # soft_weights = tf.linalg.LinearOperatorLowerTriangular(soft_weights)

            # Option 2: Subtract INF to non-lower-triangular elements
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones(soft_weights.shape[1:]))
            mask = np.inf*(mask - tf.linalg.diag(mask))
            soft_weights -= mask

        soft_weights = SoftMax(soft_weights/self.key_dim)

        return soft_weights@self.value_map(value)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, out_dim):
        super().__init__()
        self.num_heads = num_heads
        self.attention_stack = [AttentionHead(out_dim) for _ in range(num_heads)]
        self.feed_forward = LinearMap(out_dim)

    def call(self, query, key, value, causal_mask=False):
        # Parallelize if possible
        outputs = [self.attention_stack[i](query,key,value,causal_mask) for i in range(self.num_heads)]
        out_concat = tf.concat(outputs,axis=0)
        return self.feed_forward(out_concat)



