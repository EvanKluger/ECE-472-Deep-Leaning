import pytest

# MLP Class Unit Tests


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality_mlp(num_outputs):
    import tensorflow as tf

    from mlp_spiral import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_hidden_layers = 2
    hidden_layer_width = 32

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )
    input_data = tf.ones((1, num_inputs))
    output = mlp(input_data)

    tf.assert_equal(tf.shape(output), (1, num_outputs))


@pytest.mark.parametrize("hidden_layer_width", [1, 16, 128])
def test_additivity_mlp(hidden_layer_width):
    import tensorflow as tf

    from mlp_spiral import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 2

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_greater(
        abs((mlp(a) + mlp(b)) - mlp(a + b)),
        2.22e-15,
        summarize=2,
        message="MLP is passing the Additivity Condition for Linearity",
    )


@pytest.mark.parametrize("num_hidden_layers", [1, 16, 128])
def test_homogeneity_mlp(num_hidden_layers):
    import tensorflow as tf

    from mlp_spiral import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    hidden_layer_width = 32

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_greater(
        abs(((mlp(a) * b) - mlp(a * b))),
        2.22e-15,
        summarize=2,
        message="MLP is passing the Homogenity Condition for Linearity",
    )


@pytest.mark.parametrize("seed", [2384230948, 965682367, 172398674])
def test_reproduceable_mlp(seed):
    import tensorflow as tf

    from mlp_spiral import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(seed)

    num_inputs = 10
    num_outputs = 1
    num_hidden_layers = 2
    hidden_layer_width = 32

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )

    a = rng.normal(shape=[1, num_inputs])
    b = a

    tf.debugging.assert_equal(
        mlp(a),
        mlp(b),
        summarize=2,
        message="MLP is not passing the condition for Reproduceable",
    )


def test_output_range_mlp():
    import tensorflow as tf

    from mlp_spiral import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 500
    num_outputs = 1
    num_hidden_layers = 5
    hidden_layer_width = 32

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.sigmoid,
    )

    a = rng.normal(shape=[1, num_inputs])

    output = tf.cast(mlp(a), tf.float32)
    tf.debugging.assert_greater_equal(
        output, 0.0, summarize=2, message="MLP is outputting values less than 0"
    )

    tf.debugging.assert_less_equal(
        output, 1.0, summarize=2, message="MLP is outputting values greater than 1"
    )
