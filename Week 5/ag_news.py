import tensorflow as tf
from datasets import load_dataset
from tqdm import trange
from pathlib import Path
import yaml
import argparse
from transformers import AutoTokenizer, TFAutoModel


class AGNewsClassifier(tf.Module):
    def __init__(self, model_name, num_classes):
        self.bert_model = TFAutoModel.from_pretrained(model_name)

        hidden_size = self.bert_model.config.hidden_size

        self.fc = tf.Variable(
            tf.random.truncated_normal([hidden_size, hidden_size])
        )
        self.fc_bias = tf.Variable(tf.zeros([hidden_size]))

        self.dense_weights = tf.Variable(
            tf.random.truncated_normal([hidden_size, num_classes])
        )
        self.dense_bias = tf.Variable(tf.zeros([num_classes]))

    def __call__(self, inputs):
        outputs = self.bert_model(inputs)
        cls_representations = outputs[0][:, 0, :]

        fc_output = tf.nn.relu(cls_representations @ self.fc + self.fc_bias)

        logits = fc_output @ self.dense_weights + self.dense_bias
        return logits


def load_and_process_ag_news(tokenizer, split="train"):
    dataset = load_dataset("ag_news", split=split)
    encoded_dataset = dataset.map(lambda e: encode_data(e, tokenizer))
    x = tf.squeeze(tf.convert_to_tensor(encoded_dataset["input_ids"]), axis=1)
    y = tf.convert_to_tensor(encoded_dataset["label"])
    return x, y


def encode_data(example, tokenizer, max_length=256):
    encoded = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )
    return {"input_ids": encoded["input_ids"]}


def evaluate_accuracy(classifier, x, y):
    logits = classifier(x)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), dtype=tf.float32))
    return accuracy


def evaluate_test_accuracy(classifier, x_test, y_test, batch_size):
    assert (
        x_test.shape[0] % batch_size == 0
    ), f"Batch size {batch_size} is not a divisor of {x_test.shape[0]}"

    total_correct = 0
    total_samples = x_test.shape[0]

    for start_idx in range(0, total_samples, batch_size):
        end_idx = start_idx + batch_size
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        logits = classifier(x_batch)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        total_correct += tf.reduce_sum(
            tf.cast(tf.equal(predictions, y_batch), dtype=tf.int32)
        ).numpy()

    accuracy = total_correct / total_samples
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="AG News",
        description="Classify News Articles",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    tf.get_logger().setLevel("ERROR")

    num_iters = config["num_iters"]
    step_size = config["step_size"]
    batch_size = config["batch_size"]
    decay_rate = config["decay_rate"]
    refresh_rate = config["refresh_rate"]

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    x_train_full, y_train_full = load_and_process_ag_news(tokenizer, split="train")
    x_test, y_test = load_and_process_ag_news(tokenizer, split="test")

    val_size = int(0.1 * x_train_full.shape[0])

    indices = tf.random.shuffle(tf.range(x_train_full.shape[0]))

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    x_train, y_train = tf.gather(x_train_full, train_indices), tf.gather(
        y_train_full, train_indices
    )
    x_val, y_val = tf.gather(x_train_full, val_indices), tf.gather(
        y_train_full, val_indices
    )

    num_samples = x_train.shape[0]
    num_classes = 4

    classifier = AGNewsClassifier("sentence-transformers/all-MiniLM-L6-v2", num_classes)
    optimizer = tf.optimizers.Adam(learning_rate=step_size)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    best_step = 0

    bar = trange(num_iters)
    for i in bar:
        batch_indices = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=num_samples, dtype=tf.int32
        )
        x_batch = tf.gather(x_train, batch_indices)
        y_batch = tf.gather(y_train, batch_indices)

        with tf.GradientTape() as tape:
            logits = classifier(x_batch)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_batch, logits=logits
                )
            )

        grads = tape.gradient(loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
        step_size *= decay_rate

        if i % refresh_rate == 0:
            val_batch_indices = tf.random.uniform(
                shape=[1000], minval=0, maxval=x_val.shape[0], dtype=tf.int32
            )
            val_batch_x = tf.gather(x_val, val_batch_indices)
            val_batch_y = tf.gather(y_val, val_batch_indices)
            val_accuracy_batch = evaluate_accuracy(classifier, val_batch_x, val_batch_y)
            val_loss_batch = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_batch_y, logits=classifier(val_batch_x)
                )
            )

            if val_accuracy_batch > best_val_accuracy:
                best_val_accuracy = val_accuracy_batch
                best_val_loss = val_loss_batch
                best_step = i

            bar.set_description(
                f"Step {i}; Train Loss: {loss:.4f}; Val Loss: {val_loss_batch:.4f}; Val Accuracy: {val_accuracy_batch:.4f}; Step Size: {step_size:.4f}; Best Validation Accuracy: {best_val_accuracy:.4f} (at step {best_step}), Best Validation Loss: {best_val_loss:.4f}"
            )

    test_accuracy = evaluate_test_accuracy(classifier, x_test, y_test, batch_size=95)
    print(f"Test Accuracy: {test_accuracy:.4f}")
