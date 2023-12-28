import tensorflow as tf


# Basis Expansion Module Class for SGD
class BasisExpansion(tf.Module):
    def __init__(self, M, num_inputs, num_outputs):
        self.M = M
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))
        self.mu = tf.Variable(
            rng.normal(shape=[self.M, num_inputs], stddev=stddev),
            trainable=True,
            name="BasisExpansion/mu",
        )

        self.sigma = tf.Variable(
            rng.normal(shape=[self.M, num_inputs], stddev=stddev),
            trainable=True,
            name="BasisExpansion/sigma",
        )

    def __call__(self, x):
        phi = tf.exp(-tf.square(x[:, tf.newaxis] - self.mu) / (tf.square(self.sigma)))
        return tf.reduce_sum(phi, axis=2)


# Professor Curro Linear Module
class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


# Professor Curro Grad_Update Function
def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


# Professor Curro Linear Regression adpated w/ Basis Expansion for a Sin Wave
if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    import numpy as np

    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    num_inputs = 1
    num_outputs = 1

    x = rng.uniform(shape=(num_samples, 1))
    w = rng.normal(shape=(num_inputs, num_outputs))
    b = rng.normal(shape=(1, num_outputs))
    e = rng.normal(shape=(num_samples, 1), stddev=config["data"]["noise_stddev"])
    y = tf.math.sin(2 * np.pi * x) + e

    M = 6
    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)
    linear = Linear(M, num_inputs)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            phi = basis_expansion(x_batch)
            y_hat = linear(phi)

            loss = 0.5 * tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(
            loss, linear.trainable_variables + basis_expansion.trainable_variables
        )
        grad_update(
            step_size,
            linear.trainable_variables + basis_expansion.trainable_variables,
            grads,
        )

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig1, ax1 = plt.subplots()
    ax1.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x", label="Data Points")

    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax1.plot(
        a.numpy().squeeze(),
        tf.math.sin(2 * np.pi * a).numpy().squeeze(),
        label="Noiseless Sin Wave",
    )

    phi_a = basis_expansion(a)
    predictions = linear(phi_a).numpy()

    ax1.plot(
        a.numpy().squeeze(),
        tf.reshape(predictions, [-1]),
        color="red",
        linestyle="dashed",
        label="Regression Model",
    )
    ax1.legend()
    plt.title("SGD to model a Sin Wave")
    fig1.savefig("sin_wave.pdf")

    fig2, ax2 = plt.subplots()

    basis_values = basis_expansion(x)
    for j in range(M):
        ax2.plot(a.numpy(), basis_expansion(a)[:, j].numpy(), label=f"Basis {j+1}")
    plt.title("Basis Expansion Plot")
    ax2.legend()
    fig2.savefig("basis_expansion_M.pdf")
