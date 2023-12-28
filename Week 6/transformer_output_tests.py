import pytest
import tensorflow as tf
from transformer import (
    DenseLayer,
    LayerNormalization,
    MultiHeadAttention,
    TokenAndPositionEmbedding,
    TransformerBlock,
)

model_dim = 512
num_heads = 8
seq_length = 10
dff = 2048
vocab_size = 10000
max_len = 500


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_multihead_attention_output_shape(model_dim, seq_length):
    mha = MultiHeadAttention(num_heads, model_dim)
    input_data = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    output = mha(input_data)
    tf.debugging.assert_shapes([(output, (1, seq_length, model_dim))])


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_multihead_attention_non_additivity(model_dim, seq_length):
    mha = MultiHeadAttention(num_heads, model_dim)
    a = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    b = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(mha(a + b) - (mha(a) + mha(b)))),
        0.0,
    )


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_multihead_attention_non_homogeneity(model_dim, seq_length):
    mha = MultiHeadAttention(num_heads, model_dim)
    a = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    scalar = tf.constant(2.0)
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(scalar * mha(a) - mha(scalar * a))),
        0.0,
    )


@pytest.mark.parametrize(
    "seq_length, vocab_size",
    [
        (10, 10000),
        (15, 15000),
        (20, 20000),
    ],
)
def test_transformer_block_output_shape(seq_length, vocab_size):
    transformer_block = TransformerBlock(num_heads, model_dim, dff, vocab_size, max_len)
    input_data = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    output = transformer_block(input_data, mask=None)
    tf.debugging.assert_shapes([(output, (1, seq_length, vocab_size))])


@pytest.mark.parametrize(
    "seq_length, vocab_size",
    [
        (10, 10000),
        (15, 15000),
        (20, 20000),
    ],
)
def test_transformer_block_different_inputs(seq_length, vocab_size):
    transformer_block = TransformerBlock(num_heads, model_dim, dff, vocab_size, max_len)
    input_a = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    input_b = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    output_a = transformer_block(input_a, mask=None)
    output_b = transformer_block(input_b, mask=None)
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(output_a - output_b)),
        0.0,
    )


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_dense_layer_output_shape(model_dim, seq_length):
    dense_layer = DenseLayer(model_dim, dff)
    input_data = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    output = dense_layer(input_data)
    tf.debugging.assert_shapes([(output, (1, seq_length, dff))])


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_layer_norm_output_shape(model_dim, seq_length):
    layer_norm = LayerNormalization(model_dim)
    input_data = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    output = layer_norm(input_data)
    tf.debugging.assert_shapes([(output, (1, seq_length, model_dim))])


@pytest.mark.parametrize(
    "max_len, vocab_size, model_dim",
    [
        (500, 10000, 512),
        (600, 15000, 256),
        (700, 20000, 128),
    ],
)
def test_token_pos_embedding_output_shape(max_len, vocab_size, model_dim):
    token_pos_embedding = TokenAndPositionEmbedding(max_len, vocab_size, model_dim)
    input_data = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)
    output = token_pos_embedding(input_data)
    tf.debugging.assert_shapes([(output, (1, seq_length, model_dim))])


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_dense_layer_non_linearity(model_dim, seq_length):
    dense_layer = DenseLayer(model_dim, dff)
    a = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    b = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(dense_layer(a + b) - (dense_layer(a) + dense_layer(b)))),
        0.0,
    )


@pytest.mark.parametrize(
    "model_dim, seq_length",
    [
        (512, 10),
        (256, 15),
        (128, 20),
    ],
)
def test_layer_norm_scale_invariance(model_dim, seq_length):
    layer_norm = LayerNormalization(model_dim)
    a = tf.random.uniform((1, seq_length, model_dim), dtype=tf.float32)
    scalar = tf.constant(2.0)

    tf.debugging.assert_near(
        layer_norm(a),
        layer_norm(scalar * a),
        atol=1e-5,
    )
