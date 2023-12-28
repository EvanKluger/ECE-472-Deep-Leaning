import argparse
import pickle
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange


class Conv2D(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1):
        self.bias = tf.Variable(tf.zeros([output_channels]))
        stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * input_channels))
        self.kernel = tf.Variable(
            tf.random.normal(
                [kernel_size[0], kernel_size[1], input_channels, output_channels],
                mean=0,
                stddev=stddev,
            )
        )

    def __call__(self, x):
        return (
            tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
            + self.bias
        )


class Classifier(tf.Module):
    def __init__(self, input_depth, layer_depths, layer_kernel_sizes, num_classes):
        self.layers = []

        self.init_conv = Conv2D(input_depth, input_depth, layer_kernel_sizes[0])
        current_depth = input_depth

        for depth in layer_depths[:2]:
            self.layers.append(
                ResidualBlock(current_depth, depth, layer_kernel_sizes[0])
            )
            current_depth = depth

        for depth in layer_depths[1:3]:
            self.layers.append(
                ResidualBlock(current_depth, depth, layer_kernel_sizes[1])
            )
            current_depth = depth

        for depth in layer_depths[2:]:
            self.layers.append(
                ResidualBlock(current_depth, depth, layer_kernel_sizes[2])
            )
            current_depth = depth

        self.flat_size = current_depth
        self.fc_bias = tf.Variable(tf.zeros([num_classes]))
        self.fc = tf.Variable(
            tf.random.normal(
                [self.flat_size, num_classes],
                mean=0,
                stddev=np.sqrt(2.0 / self.flat_size),
            )
        )
        print(
            "Number of parameters in the last layer:",
            tf.reduce_prod(self.fc.shape).numpy(),
        )

    def __call__(self, x):
        x = self.init_conv(x)
        for layer in self.layers[:2]:
            x = layer(x)
        x = tf.nn.max_pool2d(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        for layer in self.layers[2:4]:
            x = layer(x)
        x = tf.nn.max_pool2d(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        for layer in self.layers[4:]:
            x = layer(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.nn.dropout(x, rate=0.1)
        return x @ self.fc + self.fc_bias


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


class GroupNorm(tf.Module):
    def __init__(self, num_channels, num_groups, gamma_init=1.0, beta_init=0.0):
        self.gamma = tf.Variable(tf.ones([1, num_channels, 1, 1]) * gamma_init)
        self.beta = tf.Variable(tf.zeros([1, num_channels, 1, 1]) + beta_init)
        self.num_groups = num_groups

    def __call__(self, x, eps=1e-5):
        N, H, W, C = x.shape

        G = self.num_groups
        x = tf.reshape(x, [N, G, C // G, H, W])

        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)

        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [N, H, W, C])
        gamma_reshaped = tf.reshape(self.gamma, [1, 1, 1, -1])
        beta_reshaped = tf.reshape(self.beta, [1, 1, 1, -1])
        return x * gamma_reshaped + beta_reshaped


class ResidualBlock(tf.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size, num_groups=4, stride=1
    ):
        self.conv1 = Conv2D(input_channels, output_channels, kernel_size, stride=stride)
        self.groupnorm1 = GroupNorm(output_channels, num_groups)

        self.conv2 = Conv2D(output_channels, output_channels, kernel_size)
        self.groupnorm2 = GroupNorm(output_channels, num_groups)

        self.conv3 = Conv2D(output_channels, output_channels, kernel_size)
        self.groupnorm3 = GroupNorm(output_channels, num_groups)

        self.use_projection = stride != 1 or input_channels != output_channels

        if self.use_projection:
            self.projection = Conv2D(
                input_channels, output_channels, [1, 1], stride=stride
            )
            self.groupnorm_proj = GroupNorm(output_channels, num_groups)

    def __call__(self, x):
        residual = x
        x = tf.nn.relu(self.groupnorm1(self.conv1(x)))
        x = tf.nn.relu(self.groupnorm2(self.conv2(x)))
        x = tf.nn.relu(self.groupnorm3(self.conv3(x)))

        if self.use_projection:
            residual = self.groupnorm_proj(self.projection(residual))
        return tf.nn.relu(x + residual)


def load_cifar_file(tar, file_name, dataset):
    with tar.extractfile(file_name) as file:
        data_dict = pickle.load(file, encoding="bytes")

    if dataset == "CIFAR10":
        images = (
            data_dict[b"data"]
            .reshape(-1, 3, 32, 32)
            .transpose(0, 2, 3, 1)
            .astype(np.float32)
            / 255.0
        )
        labels = np.array(data_dict[b"labels"])
    else:
        images = (
            data_dict[b"data"]
            .reshape(-1, 3, 32, 32)
            .transpose(0, 2, 3, 1)
            .astype(np.float32)
            / 255.0
        )
        labels = np.array(data_dict[b"fine_labels"])
    return images, labels


def load_and_process_cifar(filename, dataset="CIFAR10"):
    with tarfile.open(filename, "r:gz") as tar:
        x_list, y_list = [], []

        if dataset == "CIFAR10":
            for i in range(1, 6):
                x, y = load_cifar_file(
                    tar, f"cifar-10-batches-py/data_batch_{i}", dataset
                )
                x_list.append(x)
                y_list.append(y)
            x_test, y_test = load_cifar_file(
                tar, "cifar-10-batches-py/test_batch", dataset
            )
        else:
            x, y = load_cifar_file(tar, "cifar-100-python/train", dataset)
            x_list.append(x)
            y_list.append(y)
            x_test, y_test = load_cifar_file(tar, "cifar-100-python/test", dataset)

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        x_train = x[:40000]
        y_train = y[:40000]

        x_val = x[40000:]
        y_val = y[40000:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(images):
    brightness_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.18
    delta = tf.random.uniform([images.shape[0], 1, 1, 1], maxval=0.2)
    images = tf.where(
        brightness_probs, tf.image.adjust_brightness(images, delta), images
    )

    contrast_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.18
    contrast_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
    adjusted_images = tf.image.adjust_contrast(images, contrast_factor)
    images = tf.where(contrast_probs, adjusted_images, images)

    saturation_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.18
    saturation_factor = tf.random.uniform([], minval=0.8, maxval=1.2)
    adjusted_images = tf.image.adjust_saturation(images, saturation_factor)
    images = tf.where(saturation_probs, adjusted_images, images)

    hue_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.18
    hue_factor = tf.random.uniform([], minval=-0.2, maxval=0.2)
    images = tf.where(hue_probs, tf.image.adjust_hue(images, hue_factor), images)

    cutout_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.25
    offset_heights = tf.random.uniform(
        [images.shape[0]], minval=0, maxval=images.shape[1] - 8, dtype=tf.int32
    )
    offset_widths = tf.random.uniform(
        [images.shape[0]], minval=0, maxval=images.shape[2] - 8, dtype=tf.int32
    )
    mask_height = tf.sequence_mask(
        offset_heights + 8, images.shape[1]
    ) & ~tf.sequence_mask(offset_heights, images.shape[1])
    mask_width = tf.sequence_mask(
        offset_widths + 8, images.shape[2]
    ) & ~tf.sequence_mask(offset_widths, images.shape[2])
    mask_height = tf.reshape(mask_height, [images.shape[0], images.shape[1], 1])
    mask_width = tf.reshape(mask_width, [images.shape[0], 1, images.shape[2]])
    mask = tf.cast(mask_height, dtype=tf.int32) * tf.cast(mask_width, dtype=tf.int32)
    mask = 1 - tf.expand_dims(mask, axis=3)
    images_with_cutout = images * tf.cast(mask, images.dtype)
    images = tf.where(cutout_probs, images_with_cutout, images)

    crop_probs = tf.random.uniform([images.shape[0]]) < 0.25
    all_cropped_images = tf.image.random_crop(images, size=[images.shape[0], 28, 28, 3])
    all_resized_images = tf.image.resize(all_cropped_images, [32, 32])
    images = tf.where(crop_probs[:, None, None, None], all_resized_images, images)

    flip_horz_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.25
    images = tf.where(flip_horz_probs, tf.image.flip_left_right(images), images)

    flip_vert_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.01
    images = tf.where(flip_vert_probs, tf.image.flip_up_down(images), images)

    rotation_probs = tf.random.uniform([images.shape[0]]) < 0.25
    all_angles_rad = tf.random.uniform(
        [images.shape[0]], minval=-30, maxval=30, dtype=tf.float32
    ) * (np.pi / 180)
    all_rotated_images = tfa.image.rotate(images, angles=all_angles_rad)
    images = tf.where(rotation_probs[:, None, None, None], all_rotated_images, images)

    crop_resize_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.25
    cropped_resized_images = tf.image.resize(
        tf.image.random_crop(images, size=[images.shape[0], 28, 28, 3]), [32, 32]
    )
    images = tf.where(crop_resize_probs, cropped_resized_images, images)

    noise_probs = tf.random.uniform([images.shape[0], 1, 1, 1]) < 0.18
    noise = tf.random.normal(shape=images.shape, mean=0.0, stddev=0.01)
    images = tf.where(noise_probs, tf.clip_by_value(images + noise, 0, 1), images)

    images = tf.clip_by_value(images, 0, 1)

    return images


def evaluate_accuracy(classifier, x, y):
    logits = classifier(x)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y), dtype=tf.float32))
    return loss, accuracy


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

    for dataset in ["CIFAR10", "CIFAR100"]:
        if dataset == "CIFAR10":
            x_train, y_train, x_val, y_val, x_test, y_test = load_and_process_cifar(
                "cifar-10-python.tar.gz", dataset
            )
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = load_and_process_cifar(
                "cifar-100-python.tar.gz", dataset
            )

        num_samples = x_train.shape[0]
        input_depth = x_train.shape[-1]
        num_classes = 10 if dataset == "CIFAR10" else 100

        classifier = Classifier(
            input_depth, layer_depths, layer_kernel_sizes, num_classes
        )
        adam = Adam(learning_rate=step_size)
        print(
            "num_params",
            tf.math.add_n(
                [
                    tf.math.reduce_prod(var.shape)
                    for var in classifier.trainable_variables
                ]
            ),
        )
        best_val_loss = float("inf")
        best_val_acc = 0
        patience_counter = 0
        max_patience = 100
        l2_reg_strength = 0.000005

        bar = trange(num_iters)
        initial_loss, _ = evaluate_accuracy(
            classifier, x_train[:batch_size], y_train[:batch_size]
        )
        print(f"Initial Loss: {initial_loss:.4f}")

        for i in bar:
            batch_indices = rng.uniform(
                shape=[batch_size], maxval=num_samples, dtype=tf.int32
            )
            x_batch = tf.gather(x_train, batch_indices)
            y_batch = tf.gather(y_train, batch_indices)

            if dataset == "CIFAR10":
                x_batch = data_augmentation(x_batch)

            if dataset == "CIFAR100" and i % 3 == 0:
                x_batch = data_augmentation(x_batch)

            with tf.GradientTape() as tape:
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

                adam.apply_gradients(grads, classifier.trainable_variables)
                step_size *= decay_rate
                loss_float = float(loss.numpy().mean())

                val_logits = classifier(x_val)
                val_predicted_probs = tf.nn.softmax(val_logits).numpy()
                val_top5_accuracy = top_k_accuracy_score(
                    y_val, val_predicted_probs, k=5
                )
                val_loss, val_accuracy = evaluate_accuracy(classifier, x_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy

                bar.set_description(
                    f"Dataset: {dataset}; Step {i}; Train Loss: {loss_float:.4f}; Val Loss: {val_loss:.4f}; Val Accuracy: {val_accuracy:.4f}; Best Val Loss: {best_val_loss:.4f}; Best Val Acc: {best_val_acc:.4f}; Val Top-5 Accuracy: {val_top5_accuracy * 100:.2f}%; Step Size: {step_size:.4f}"
                )
                bar.refresh()

        test_loss, test_accuracy = evaluate_accuracy(classifier, x_test, y_test)
        test_logits = classifier(x_test)
        test_predicted_probs = tf.nn.softmax(test_logits).numpy()
        test_top5_accuracy = top_k_accuracy_score(y_test, test_predicted_probs, k=5)
        test_top3_accuracy = top_k_accuracy_score(y_test, test_predicted_probs, k=3)

        print(
            f"\nDataset: {dataset}; Test Loss: {test_loss:.4f}; Test Accuracy: {test_accuracy:.4f}; Test Top-3 Accuracy: {test_top3_accuracy * 100:.2f}%; Test Top-5 Accuracy: {test_top5_accuracy * 100:.2f}%"
        )
