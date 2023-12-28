import numpy as np
import tensorflow as tf

from transformer import TransformerBlock


def compute_jacobian(inputs, outputs):
    num_inputs = tf.size(inputs).numpy()
    num_outputs = tf.size(outputs).numpy()

    jacobian = np.zeros((num_outputs, num_inputs))

    for i in range(num_outputs):
        grad = tf.gradients(outputs[i], inputs)[0]
        if grad is not None:
            jacobian[i] = grad.numpy().flatten()

    return jacobian


model_dim = 512
num_heads = 8
seq_length = 10
dff = 2048
vocab_size = 10000
max_len = 500


transformer_block = TransformerBlock(num_heads, model_dim, dff, vocab_size, max_len)

batch_size = 1
sequence_length = 5
inputs = tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32)

with tf.GradientTape(persistent=True) as tape:
    outputs = transformer_block(inputs)

    embedding_outputs = transformer_block.embedding_output
    tape.watch(embedding_outputs)


jacobian_matrix = np.zeros((batch_size, sequence_length, model_dim))

Fail = False

for i in range(sequence_length):
    grad = tape.gradient(outputs[:, i], embedding_outputs)
    if grad is not None:
        jacobian_matrix[:, i, :] = grad.numpy()

for i in range(sequence_length):
    for j in range(i + 1, sequence_length):
        if not np.allclose(jacobian_matrix[:, i, j:], 0, atol=1e-6):
            Fail = True
            print("The test for the casual masking property of the Transformer Failed")
