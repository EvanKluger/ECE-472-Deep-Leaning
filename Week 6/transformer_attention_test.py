import tensorflow as tf

from transformer import MultiHeadAttention


def test_attention():
    num_heads = 8
    model_dim = 512

    mha = MultiHeadAttention(num_heads, model_dim)

    a = tf.random.uniform((1, 60, model_dim), dtype=tf.float32)
    b = tf.random.uniform((1, 60, model_dim), dtype=tf.float32)

    attention_1 = mha(a)
    attention_2 = mha(b)
    c = tf.identity(a)
    attention_3 = mha(c)

    tf.debugging.assert_less(
        0.0,
        tf.reduce_max(tf.abs(attention_1 - attention_2)),
    )
    tf.debugging.assert_near(
        attention_1,
        attention_3,
    )
