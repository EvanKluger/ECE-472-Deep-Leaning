import pytest
import tensorflow as tf

from cifar import Classifier, Conv2D, GroupNorm, ResidualBlock


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5]), (1, 8, [3, 3])],
)
def test_conv2d_shape(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)
    input_data = tf.ones((1, 32, 32, input_channels))
    output = conv_layer(input_data)

    expected_shape = (1, 32, 32, output_channels)
    tf.debugging.assert_shapes([(output, expected_shape)])


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5]), (1, 8, [3, 3])],
)
def test_conv2d_additivity(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)

    a = tf.ones((1, 32, 32, input_channels))
    b = tf.ones((1, 32, 32, input_channels))

    tf.debugging.assert_near(
        (conv_layer(a) + conv_layer(b)),
        conv_layer(a + b),
        atol=1e-5,
        message="Conv2D is not behaving Linearity",
    )


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5]), (1, 8, [3, 3])],
)
def test_conv2d_homogeneity(input_channels, output_channels, kernel_size):
    conv_layer = Conv2D(input_channels, output_channels, kernel_size)

    a = tf.ones((1, 32, 32, input_channels))
    b = 2 * a

    tf.debugging.assert_near(
        2 * conv_layer(a),
        conv_layer(b),
        atol=1e-5,
        message="Conv2D is not behavingr Linearity",
    )


@pytest.mark.parametrize(
    "input_depth, layer_depths, layer_kernel_sizes, num_classes, expected_shape",
    [
        (3, [16, 32], [[3, 3], [3, 3]], 10, (10, 10)),
        (3, [32, 64], [[3, 3], [3, 3]], 100, (10, 100)),
    ],
)
def test_classifier_output_shape(
    input_depth, layer_depths, layer_kernel_sizes, num_classes, expected_shape
):
    classifier = Classifier(input_depth, layer_depths, layer_kernel_sizes, num_classes)
    input_data = tf.ones((10, 32, 32, input_depth))
    output = classifier(input_data)

    tf.debugging.assert_shapes([(output, expected_shape)])


@pytest.mark.parametrize(
    "num_channels, num_groups",
    [(16, 2), (32, 4)],
)
def test_GroupNorm_output_shape(num_channels, num_groups):
    groupnorm_layer = GroupNorm(num_channels, num_groups)
    input_data = tf.ones((1, 32, 32, num_channels))
    output = groupnorm_layer(input_data)

    expected_shape = (1, 32, 32, num_channels)
    tf.debugging.assert_shapes([(output, expected_shape)])


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5])],
)
def test_ResidualBlock_non_additivity(input_channels, output_channels, kernel_size):
    residualblock = ResidualBlock(input_channels, output_channels, kernel_size)

    a = tf.random.normal((1, 32, 32, input_channels))
    b = tf.ones((1, 32, 32, input_channels))

    tf.debugging.assert_greater(
        tf.reduce_max(
            tf.abs(residualblock(a + b) - (residualblock(a) + residualblock(b)))
        ),
        0.0,
        summarize=2,
        message="ResidualBlock is not linear as expected - doesn't pass the Additivity Condition",
    )


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5])],
)
def test_ResidualBlock_non_homogeneity(input_channels, output_channels, kernel_size):
    residualblock = ResidualBlock(input_channels, output_channels, kernel_size)

    a = tf.random.normal((1, 32, 32, input_channels))
    b = 2 * a
    print(2 * residualblock(a))
    print(residualblock(b))
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(2 * residualblock(a) - residualblock(b))),
        0.0,
        summarize=2,
        message="ResidualBlock is acting linear",
    )


@pytest.mark.parametrize(
    "input_channels, output_channels, kernel_size",
    [(3, 16, [3, 3]), (16, 32, [5, 5])],
)
def test_ResidualBlock_output_shape(input_channels, output_channels, kernel_size):
    residualblock = ResidualBlock(input_channels, output_channels, kernel_size)
    input_data = tf.ones((1, 32, 32, input_channels))
    output = residualblock(input_data)

    expected_shape = (1, 32, 32, output_channels)
    tf.debugging.assert_shapes([(output, expected_shape)])
