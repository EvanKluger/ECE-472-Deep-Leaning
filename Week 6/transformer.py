import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.Module):
    def __init__(self, num_heads, model_dim):
        self.num_heads = num_heads
        self.model_dim = model_dim
        assert model_dim % self.num_heads == 0
        self.depth = model_dim // self.num_heads

        self.query = tf.Variable(tf.random.normal([model_dim, model_dim]))
        self.key = tf.Variable(tf.random.normal([model_dim, model_dim]))
        self.value = tf.Variable(tf.random.normal([model_dim, model_dim]))
        self.output = tf.Variable(tf.random.normal([model_dim, model_dim]))

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention_score(self, Q, K, V, mask=None):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += tf.cast(mask, tf.float32) * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, V)

        return output

    def generate_causal_mask(self, seq_len, batch_size, num_heads):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, ...]
        return mask * tf.ones([batch_size, num_heads, 1, 1])

    def __call__(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        x = tf.cast(x, dtype=tf.float32)

        Q = tf.matmul(x, self.query)
        K = tf.matmul(x, self.key)
        V = tf.matmul(x, self.value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        mask = self.generate_causal_mask(seq_len, batch_size, self.num_heads)

        attention_output = self.attention_score(Q, K, V, mask)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        concatenated_output = tf.reshape(
            attention_output, (batch_size, -1, self.model_dim)
        )

        return tf.matmul(concatenated_output, self.output)


class TransformerBlock(tf.Module):
    def __init__(self, num_heads, d_model, dff, vocab_size, max_len):
        self.embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, d_model)
        self.embedding_output = None

        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)

        self.ffn_input = DenseLayer(d_model, dff, activation="relu")
        self.ffn_output = DenseLayer(dff, d_model)

        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)

        self.dropout1 = tf.nn.dropout
        self.dropout2 = tf.nn.dropout

        self.final_layer = DenseLayer(d_model, vocab_size)

    def feed_forward_network(self, x):
        x = self.ffn_input(x)
        return self.ffn_output(x)

    def __call__(self, x, mask=None):
        x = self.embedding_layer(x)
        self.embedding_output = x

        attn_output = self.multi_head_attention(x, mask)

        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.feed_forward_network(out1)

        out2 = self.layernorm2(out1 + ffn_output)

        final_output = self.final_layer(out2)

        return final_output

    @property
    def trainable_variables(self):
        return (
            self.multi_head_attention.trainable_variables
            + self.ffn_input.trainable_variables
            + self.ffn_output.trainable_variables
            + self.layernorm1.trainable_variables
            + self.layernorm2.trainable_variables
        )


class DenseLayer(tf.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        std_dev = np.sqrt(2 / input_dim)
        self.weights = tf.Variable(
            tf.random.normal([input_dim, output_dim], stddev=std_dev)
        )
        self.bias = tf.Variable(tf.zeros([output_dim]))
        self.activation = activation

    def __call__(self, x):
        z = tf.matmul(x, self.weights) + self.bias
        if self.activation == "relu":
            return tf.nn.relu(z)
        return z


class LayerNormalization(tf.Module):
    def __init__(self, d_model, epsilon=1e-6):
        self.scale = tf.Variable(tf.ones([d_model]))
        self.bias = tf.Variable(tf.zeros([d_model]))
        self.epsilon = epsilon

    def __call__(self, x):
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.bias


class TokenAndPositionEmbedding(tf.Module):
    def __init__(self, max_len, vocab_size, model_dim):
        self.token_embeddings = tf.Variable(tf.random.normal([vocab_size, model_dim]))
        self.position_embeddings = tf.Variable(tf.random.normal([max_len, model_dim]))

    def __call__(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.position_embeddings[:length, :]
        token_embeddings = tf.nn.embedding_lookup(params=self.token_embeddings, ids=x)
        return token_embeddings + positions
