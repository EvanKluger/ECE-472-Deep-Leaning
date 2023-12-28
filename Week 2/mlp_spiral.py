import numpy as np
import tensorflow as tf

from linear import Linear


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):
        self.layers = []

        self.layers.append((Linear(num_inputs, hidden_layer_width), hidden_activation))

        for _ in range(1, num_hidden_layers):
            hidden_layer = Linear(hidden_layer_width, hidden_layer_width)
            self.layers.append((hidden_layer, hidden_activation))

        self.output_layer = (Linear(hidden_layer_width, num_outputs), output_activation)

    def __call__(self, x):
        for layer, activation in self.layers:
            x = activation(layer(x))
        output_linear, output_activation = self.output_layer
        return output_activation(output_linear(x))


# Function to generete spiral data
def generate_spiral_data(num_samples, noise):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    e = rng.normal(shape=(num_samples,), stddev=noise)
    r = tf.linspace(3.14 / 8, 4 * 3.14, num_samples)
    x1 = (r + e) * tf.math.cos(r)
    y1 = (r + e) * tf.math.sin(r)
    X1 = tf.stack([x1, y1], axis=1)
    x2 = (r + e) * (-1) * tf.math.cos(r)
    y2 = (r + e) * (-1) * tf.math.sin(r)
    X2 = tf.stack([x2, y2], axis=1)

    X = tf.concat([X1, X2], axis=0)
    y = tf.concat(
        [tf.zeros(num_samples, dtype=tf.uint8), tf.ones(num_samples, dtype=tf.uint8)],
        axis=0,
    )

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.shuffle(buffer_size=2 * num_samples)

    X_shuffled, y_shuffled = zip(*list(dataset.as_numpy_iterator()))
    X_shuffled = tf.stack(X_shuffled, axis=0)
    y_shuffled = tf.stack(y_shuffled, axis=0)

    return X_shuffled, y_shuffled


# Professor Curro Grad_Update Function
def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from sklearn.inspection import DecisionBoundaryDisplay
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
    noise = config["data"]["noise"]

    x, y = generate_spiral_data(num_samples, noise)

    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 6
    hidden_layer_width = 200
    l2_reg_strength = 0.000001

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )

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

            y_hat = mlp(x_batch)
            y_batch = tf.cast(tf.expand_dims(y_batch, -1), tf.float32)

            loss = -tf.reduce_mean(
                y_batch * tf.math.log(y_hat) + (1 - y_batch) * tf.math.log(1 - y_hat)
            )
            l2_loss = tf.reduce_sum(mlp.layers[0][0].w ** 2)

            for layer, _ in mlp.layers[1:]:
                l2_loss += tf.reduce_sum(layer.w ** 2)

            loss += l2_reg_strength * l2_loss
        grads = tape.gradient(loss, mlp.trainable_variables)
        grad_update(step_size, mlp.trainable_variables, grads)
        step_size *= decay_rate
        loss_float = float(loss.numpy().mean())

        y_hat_binary = tf.cast(y_hat > 0.5, dtype=tf.float32)
        correct_classification = tf.reduce_all(tf.equal(y_batch, y_hat_binary))

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss_float:0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch, y_hat_binary), tf.float32))
    print(f"Final Accuracy of Last Batch: {accuracy.numpy():.4f}")

    x_min, x_max = tf.reduce_min(x[:, 0]) - 1, tf.reduce_max(x[:, 0]) + 1
    y_min, y_max = tf.reduce_min(x[:, 1]) - 1, tf.reduce_max(x[:, 1]) + 1

    xx0, xx1 = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    Z = mlp(np.c_[xx0.ravel(), xx1.ravel()])
    Z = Z.numpy().reshape(xx0.shape)

    display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=Z)
    display.plot(
        cmap="jet",
        alpha=0.8,
    )

    plt.scatter(
        x[:, 0], x[:, 1], c=tf.squeeze(y).numpy(), cmap="jet", edgecolor="k", s=20
    )
    plt.title("Decision Boundary of Two Spiral Plots")
    plt.savefig("two_spiral_decision_boundary.pdf")
