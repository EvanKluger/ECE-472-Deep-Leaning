import pytest


# Tests for The Linear Class taken from Professor Curro's linear_tests.py
# Plus Some Additional Tests Not Included in linear_tests.py
def test_additivity_Linear():
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(
        linear(a + b),
        linear(a) + linear(b),
        summarize=2,
        message="Linear is failing the Additivity Condition for Linearity",
    )


def test_homogeneity_Linear():
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(
        linear(a * b),
        linear(a) * b,
        summarize=2,
        message="Linear is failing the Homogeneity Condition for Linearity",
    )


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality_Linear(num_outputs):
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = linear(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable_Linear(bias):
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs, bias=bias)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, linear.trainable_variables)

    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_init_properties_Linear(a_shape, b_shape):
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape

    linear_a = Linear(num_inputs_a, num_outputs_a, bias=False)
    linear_b = Linear(num_inputs_b, num_outputs_b, bias=False)

    std_a = tf.math.reduce_std(linear_a.w)
    std_b = tf.math.reduce_std(linear_b.w)

    tf.debugging.assert_less(std_a, std_b)


def test_bias_Linear():
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    linear_with_bias = Linear(1, 1, bias=True)
    assert hasattr(linear_with_bias, "b")

    linear_with_bias = Linear(1, 1, bias=False)
    assert not hasattr(linear_with_bias, "b")


def test_reproduceable_Linear():
    import tensorflow as tf

    from SinGraph import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = a

    tf.debugging.assert_near(
        linear(a),
        linear(b),
        summarize=2,
    )


# Tests for The BasisExpansion Class implemented based on
# Professor Curro's Linear Class tests from linear_tests.py
def test_additivity_BasisExpansion():
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 6

    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_greater(
        tf.abs(basis_expansion(a + b) - (basis_expansion(a) + basis_expansion(b))),
        2.22e-15,
        message="BasisExpansion is acting linear",
    )


def test_homogeneity_BasisExpansion():
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 10
    num_test_cases = 100

    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_greater(
        tf.abs(basis_expansion(a * b) - (basis_expansion(a) * b)),
        2.22e-15,
        message="BasisExpansion is acting linear",
    )


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality_BasisExpansion(num_outputs):
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    M = 5

    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = basis_expansion(a)

    print(tf.shape(z))
    print(num_outputs)

    tf.assert_equal(tf.shape(z)[-1], M)


def test_trainable_BasisExpansion():
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 6

    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = basis_expansion(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, basis_expansion.trainable_variables)

    assert len(grads) == len(basis_expansion.trainable_variables)
    assert len(grads) == 2


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([5, 1000, 1000], [5, 100, 100]),
        ([6, 1000, 100], [6, 100, 100]),
        ([7, 100, 1000], [7, 100, 100]),
    ],
)
def test_init_properties_BasisExpansion(a_shape, b_shape):
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    M = 6
    _, num_inputs_a, num_outputs_a = a_shape
    _, num_inputs_b, num_outputs_b = b_shape

    basis_expansion_a = BasisExpansion(M, num_inputs_a, num_outputs_a)
    basis_expansion_b = BasisExpansion(M, num_inputs_b, num_outputs_b)

    std_mu_a = tf.math.reduce_std(basis_expansion_a.mu)
    std_mu_b = tf.math.reduce_std(basis_expansion_b.mu)

    tf.debugging.assert_less(std_mu_a, std_mu_b)


def test_reproduceable_BasisExpansion():
    import tensorflow as tf

    from SinGraph import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 5

    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = a

    tf.debugging.assert_near(
        basis_expansion(a),
        basis_expansion(b),
        summarize=2,
    )


# Integrated Tests for The BasisExpansion Class and Linear Class
# TODO: Fix these Units Tests
def test_additivity_Linear_and_BasisExpansion():
    import tensorflow as tf

    from SinGraph import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 5

    linear = Linear(M, num_inputs)
    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(
        linear(basis_expansion(a)) + linear(basis_expansion(b)),
        linear(basis_expansion(a) + basis_expansion(b)),
        summarize=2,
    )


def test_homogeneity_Linear_BasisExpansion():
    import tensorflow as tf

    from SinGraph import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100
    M = 5

    linear = Linear(M, num_inputs)
    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1, 1])

    tf.debugging.assert_near(
        linear(basis_expansion(a) * b),
        linear(basis_expansion(a)) * b,
        summarize=2,
    )


def test_reproduceable_Linear_BasisExpansion():
    import tensorflow as tf

    from SinGraph import Linear, BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 5

    linear = Linear(M, num_inputs)
    basis_expansion = BasisExpansion(M, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = a

    tf.debugging.assert_near(
        linear(basis_expansion(a)), linear(basis_expansion(b)), summarize=2
    )
