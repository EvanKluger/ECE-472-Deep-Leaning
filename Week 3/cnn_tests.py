import pytest
import tensorflow as tf

from cnn import Classifier, Conv2D


# Unit Tests for Classifier Class
@pytest.mark.parametrize("input_depth, num_outputs", [(28, 7), (28, 14), (28, 4)])
def test_Classifier_dimensionality(input_depth, num_outputs):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    layer_depths = [30, 50, 70]
    layer_kernel_sizes = [[3, 3], [3, 3], [3, 3]]

    cnn = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_outputs)

    input_data = tf.ones((1, 28, 28, input_depth))
    output = cnn(input_data)

    tf.assert_equal(tf.shape(output), (1, num_outputs))


@pytest.mark.parametrize(
    "input_depth, layer_depths", [(28, [28, 16, 32]), (28, [28, 128, 256])]
)
def test_Classifier_additivity(input_depth, layer_depths):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_outputs = 1
    layer_kernel_sizes = [[3, 3], [3, 3], [3, 3]]

    cnn = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_outputs)

    a = rng.normal(shape=[1, 28, 28, input_depth])
    b = rng.normal(shape=[1, 28, 28, input_depth])

    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs((cnn(a) + cnn(b)) - cnn(a + b))),
        2.22e-15,
        summarize=2,
        message="CNN is passing the Additivity Condition for Linearity",
    )


@pytest.mark.parametrize("layer_depths", [[30, 50], [70, 90]])
def test_Classifier_homogeneity(layer_depths):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_depth = 28
    num_outputs = 1
    layer_kernel_sizes = [[3, 3], [3, 3]]

    cnn = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_outputs)

    a = rng.normal(shape=[1, 28, 28, input_depth])
    b = 2 * a

    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(cnn(b) - (2 * cnn(a)))),
        2.22e-15,
        summarize=2,
        message="CNN is passing the Homogeneity Condition for Linearity",
    )


@pytest.mark.parametrize("seed", [2384230948, 965682367, 172398674])
def test_Classifier_reproducible_cnn(seed):
    import numpy as np

    from cnn import Conv2D

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(seed)

    input_depth = 28
    num_outputs = 1
    layer_depths = [30, 50, 70]
    layer_kernel_sizes = [[3, 3], [3, 3], [3, 3]]

    # Custom Classifier class neeeded with dropout set to 0 for reproduceability to work
    # WIthout dropout = 0. The class cannot be reproduceable
    # Note: dropout was not added to the Classifier inputs because the assignment
    # paramters specifically outlined the inputs as - input_depth, layer_depths, layer_kernel_sizes, num_classes
    # Everything abou tthis class is the same except for the dropout set to 0

    class Classifier_no_dropout(tf.Module):
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
                x = tf.nn.dropout(x, rate=0.0)
            x = tf.reshape(x, [-1, self.fc.shape[0]])
            return tf.matmul(x, self.fc) + self.fc_bias

    cnn = Classifier_no_dropout(
        input_depth, layer_depths, layer_kernel_sizes, num_outputs
    )

    a = rng.normal(shape=[1, 28, 28, input_depth])
    b = a

    tf.debugging.assert_near(
        cnn(a),
        cnn(b),
        summarize=2,
        message="CNN is not passing the condition for Reproducibility",
    )


@pytest.mark.parametrize(
    "layer_depths, layer_kernel_sizes, num_outputs",
    [
        ([16, 32, 64], [[3, 3], [3, 3], [3, 3]], 10),
        ([32, 64, 128], [[3, 3], [3, 3], [3, 3]], 5),
        ([16, 32], [[3, 3], [3, 3]], 3),
    ],
)
def test_Classifier_output_shape_hyperparameters(
    layer_depths, layer_kernel_sizes, num_outputs
):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    cnn = Classifier(28, layer_depths, layer_kernel_sizes, num_outputs)

    input_data = tf.ones((1, 28, 28, 28))
    output = cnn(input_data)
    expected_output_shape = (1, num_outputs)
    tf.debugging.assert_shapes([(output, expected_output_shape)])


# Unit Tests for Conv2 Class
@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, (3, 3)), (16, 32, (5, 5)), (1, 8, (3, 3))],
)
def test_conv2d_shape(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)
    input_data = tf.ones((1, 28, 28, input_channels))
    output = conv_layer(input_data)

    expected_shape = (1, 28, 28, output_channels)
    tf.debugging.assert_shapes([(output, expected_shape)])


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, (3, 3)), (16, 32, (5, 5)), (1, 8, (3, 3))],
)
def test_conv2d_additivity(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)

    a = tf.ones((1, 28, 28, input_channels))
    b = tf.ones((1, 28, 28, input_channels))

    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs((conv_layer(a) + conv_layer(b)) - conv_layer(a + b))),
        2.22e-15,
        summarize=2,
        message="CNN is passing the Additivity Condition for Linearity",
    )


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, (3, 3)), (16, 32, (5, 5)), (1, 8, (3, 3))],
)
def test_conv2d_homogeinity(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)

    a = tf.ones((1, 28, 28, input_channels))
    b = 2 * a

    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs((2 * conv_layer(a)) - conv_layer(b))),
        2.22e-15,
        summarize=2,
        message="CNN is passing the Additivity Condition for Linearity",
    )
