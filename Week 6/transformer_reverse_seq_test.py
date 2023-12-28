import numpy as np
import tensorflow as tf
from tqdm import trange
from transformer import TransformerBlock

from common_functions import Adam, plot_results, sparse_categorical_crossentropy

batch_size = 64
vocab_size = 100
seq_len = 50
dataset_size = 10000
num_iters = 50
d_model = 512
num_heads = 8
dff = 2048
decay_rate = 0.99
test_dataset_size = 1000


def generate_data(vocab_size, seq_len, dataset_size):
    data = np.random.randint(1, vocab_size, size=(dataset_size, seq_len))
    targets = np.flip(data, axis=1)
    return data, targets


def load_dataset(batch_size, vocab_size, seq_len, dataset_size):
    data, targets = generate_data(vocab_size, seq_len, dataset_size)
    dataset = tf.data.Dataset.from_tensor_slices((data, targets))
    dataset = dataset.cache()
    dataset = dataset.shuffle(dataset_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_transformer(transformer, dataset, num_iters, decay_rate):
    optimizer = Adam()
    loss_history = []
    iteration_history = []

    bar = trange(num_iters, desc="Training Reverse Seq Transformer")

    for iteration in bar:
        total_loss = 0
        batches = 0

        for inp, tar in dataset:
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            with tf.GradientTape() as tape:
                predictions = transformer(inp, tar_inp)
                predictions_sliced = predictions[:, :-1, :]
                predictions_reshaped = predictions_sliced

                tar_real_reshaped = tf.reshape(tar_real, [-1, 49])
                loss = sparse_categorical_crossentropy(
                    tar_real_reshaped, predictions_reshaped
                )

            grads = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(grads, transformer.trainable_variables)

            total_loss += loss.numpy()
            batches += 1

        optimizer.learning_rate *= decay_rate

        avg_loss = total_loss / batches
        iteration_history.append(iteration)
        loss_history.append(avg_loss)
        bar.set_description(
            f"Iter {iteration + 1}, Loss: {avg_loss:.4f}, LR: {optimizer.learning_rate:.6f}"
        )

        bar.refresh()
    plot_results(iteration_history, loss_history)
    return transformer


def test_transformer(transformer, test_dataset):
    total_accuracy = 0
    batches = 0

    for inp, tar in test_dataset:
        predictions = transformer(inp)
        predictions_sliced = predictions[:, :-1, :]
        predictions_flat = tf.argmax(predictions_sliced, axis=-1)
        tar_flat = tar[:, 1:]

        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions_flat, tar_flat), tf.float32)
        )
        total_accuracy += accuracy
        batches += 1

    avg_accuracy = total_accuracy / batches
    print(f"Test Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy


def main():
    dataset = load_dataset(batch_size, vocab_size, seq_len, dataset_size)
    transformer = TransformerBlock(num_heads, d_model, dff, vocab_size, seq_len)
    trained_transformer = train_transformer(transformer, dataset, num_iters, decay_rate)
    test_dataset = load_dataset(batch_size, vocab_size, seq_len, test_dataset_size)
    test_transformer(trained_transformer, test_dataset)


main()
