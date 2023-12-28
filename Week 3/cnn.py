import argparse
import gzip
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange


class Conv2D(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * input_channels))
        self.kernel = tf.Variable(
            tf.random.normal(
                [kernel_size[0], kernel_size[1], input_channels, output_channels],
                mean=0,
                stddev=stddev,
            )
        )
        self.bias = tf.Variable(tf.fill([output_channels], 0.00))

    def __call__(self, x):
        return (
            tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
            + self.bias
        )


class Classifier(tf.Module):
    def __init__(self, input_depth, layer_depths, layer_kernel_sizes, num_classes):
        self.layers = []

        current_depth = input_depth
        for depth, kernel_size in zip(layer_depths, layer_kernel_sizes):
            self.layers.append(Conv2D(current_depth, depth, kernel_size))
            current_depth = depth

        stddev = np.sqrt(2.0 / (current_depth * 28 * 28))
        self.fc = tf.Variable(
            tf.random.normal(
                [current_depth * 28 * 28, num_classes], mean=0, stddev=stddev
            )
        )
        self.fc_bias = tf.Variable(tf.zeros([num_classes]))

    def __call__(self, x):
        for layer in self.layers:
            x = tf.nn.relu(layer(x))
            x = tf.nn.dropout(x, rate=0.1)
        x = tf.reshape(x, [-1, self.fc.shape[0]])
        return tf.matmul(x, self.fc) + self.fc_bias


# Adam Class taken from Tensorflow Documentation
class Adam:
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t = 1.0
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        for i, (grad, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(
                self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * grad
            )
            self.s_dvar[i].assign(
                self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(grad)
            )
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2**self.t))
            var.assign_sub(
                self.learning_rate * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.epsilon))
            )
        self.t += 1.0
        return


# Class evaulate_accuracy takes in the classifier, the x data, and y labels and returns the loss and accuracy
def evaluate_accuracy(classifier, x, y):
    logits = classifier(x)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), dtype=tf.float32))
    return loss, accuracy


# Function: Loads the MNIST Images from Local File
def load_mnist_images(filename):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28, 1)
    return data / np.float32(255)


# Function: Loads the MNIST Lavbels from Local File
def load_mnist_labels(filename):
    with gzip.open(filename, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CNN",
        description="Convolutional Neural Network",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    layer_depths = config["network"]["layer_depths"]
    layer_kernel_sizes = config["network"]["layer_kernel_sizes"]
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    batch_size = config["learning"]["batch_size"]
    decay_rate = config["learning"]["decay_rate"]
    refresh_rate = config["display"]["refresh_rate"]

    x_train = load_mnist_images("train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels("train-labels-idx1-ubyte.gz").astype(np.int32)

    x_test = load_mnist_images("t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

    val_split = 1 / 6
    val_count = int(val_split * len(x_train))
    x_val, y_val = x_train[:val_count], y_train[:val_count]
    x_train, y_train = x_train[val_count:], y_train[val_count:]

    num_samples = x_train.shape[0]
    input_depth = x_train.shape[-1]
    num_classes = 10

    layer_depths = config["network"]["layer_depths"]
    layer_kernel_sizes = config["network"]["layer_kernel_sizes"]
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    batch_size = config["learning"]["batch_size"]
    refresh_rate = config["display"]["refresh_rate"]

    classifier = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_classes)
    l2_reg_strength = 0.000005

    adam = Adam(learning_rate=step_size)

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x_train, batch_indices)
            y_batch = tf.gather(y_train, batch_indices)
            logits = classifier(x_batch)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_batch, logits=logits
                )
            )

            l2_loss = 0.0
            for layer in classifier.layers:
                if isinstance(layer, Conv2D):
                    l2_loss += tf.reduce_sum(tf.square(layer.kernel))

            loss += l2_reg_strength * l2_loss
            grads = tape.gradient(loss, classifier.trainable_variables)

            new_grads = [
                grad + 2.0 * l2_reg_strength * var if "kernel" in var.name else grad
                for grad, var in zip(grads, classifier.trainable_variables)
            ]

            adam.apply_gradients(new_grads, classifier.trainable_variables)
            step_size *= decay_rate
            loss_float = float(loss.numpy().mean())
            if i % refresh_rate == (refresh_rate - 1):
                val_loss, val_accuracy = evaluate_accuracy(classifier, x_val, y_val)
                bar.set_description(
                    f"Step {i}; Train Loss: {loss_float:.4f}; Val Loss: {val_loss:.4f}; Val Accuracy: {val_accuracy:.4f}; Step Size: {step_size:.4f}"
                )
                bar.refresh()
    test_loss, test_accuracy = evaluate_accuracy(classifier, x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}; Test Accuracy: {test_accuracy:.4f}")
