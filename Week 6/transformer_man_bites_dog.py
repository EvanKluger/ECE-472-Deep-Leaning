import numpy as np
import tensorflow as tf
from tqdm import trange
from transformer import TransformerBlock

from common_functions import Adam, plot_results, sparse_categorical_crossentropy


vocab = {"man": 1, "bites": 2, "dog": 3, "pad": 0}
vocab_size = len(vocab)

batch_size = 64
vocab_size = len(vocab)
seq_len = 3
dataset_size = 10000
num_iters = 10
d_model = 512
num_heads = 8
dff = 2048
decay_rate = 0.99


def test_man_bites_dog(transformer, vocab):
    input_sequence = [vocab["man"], vocab["bites"]]
    input_padded = np.array(input_sequence + [vocab["pad"]])[np.newaxis, :]
    prediction = transformer(input_padded)
    predicted_token = np.argmax(prediction[0, 1, :], axis=-1)
    predicted_id = int(predicted_token)

    predicted_word = (
        [word for word, idx in vocab.items() if idx == predicted_id][0]
        if predicted_id in vocab.values()
        else "unknown"
    )

    print(
        f'The Transformer predicts {predicted_word} as the next word after "man bites"'
    )


def generate_man_bites_dog_data(vocab, dataset_size):
    data = np.array(
        [[vocab["man"], vocab["bites"], vocab["dog"]] for _ in range(dataset_size)]
    )
    targets = np.array(
        [[vocab["bites"], vocab["dog"], vocab["pad"]] for _ in range(dataset_size)]
    )
    return data, targets


def load_man_bites_dog_dataset(batch_size, dataset_size):
    data, targets = generate_man_bites_dog_data(vocab, dataset_size)
    dataset = tf.data.Dataset.from_tensor_slices((data, targets))
    dataset = (
        dataset.cache()
        .shuffle(dataset_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def train_transformer_mbd(transformer, dataset, num_iters, decay_rate):
    optimizer = Adam()
    loss_history = []
    iteration_history = []
    bar = trange(num_iters, desc="Training Transformer")

    for iteration in bar:
        total_loss = 0
        batches = 0
        for inp, tar in dataset:
            with tf.GradientTape() as tape:
                predictions = transformer(inp)
                predictions_sliced = predictions[:, 1:, :]
                loss = sparse_categorical_crossentropy(tar[:, 1:], predictions_sliced)
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


def main():
    dataset = load_man_bites_dog_dataset(batch_size, dataset_size)
    transformer = TransformerBlock(num_heads, d_model, dff, vocab_size, seq_len)
    trained_transformer = train_transformer_mbd(
        transformer, dataset, num_iters, decay_rate
    )
    test_man_bites_dog(trained_transformer, vocab)


main()
