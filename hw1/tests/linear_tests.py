# modified from example provided by Prof. Curro

import numpy as np
import pytest


def test_additivity():
    import tensorflow as tf

    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = linear(a)

    print(z.shape)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from linear import Linear

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
def test_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape

    linear_a = Linear(num_inputs_a, num_outputs_a, bias=False)
    linear_b = Linear(num_inputs_b, num_outputs_b, bias=False)

    std_a = tf.math.reduce_std(linear_a.w)
    std_b = tf.math.reduce_std(linear_b.w)

    tf.debugging.assert_less(std_a, std_b)


def test_bias():
    import tensorflow as tf

    from linear import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    linear_with_bias = Linear(1, 1, bias=True)
    assert hasattr(linear_with_bias, "b")

    linear_with_bias = Linear(1, 1, bias=False)
    assert not hasattr(linear_with_bias, "b")


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("num_inputs", [2, 8])
@pytest.mark.parametrize("num_samples", [32, 128])
def test_integration(bias, num_inputs, num_samples):
    import tensorflow as tf
    from tqdm import trange

    from basis_expansion import BasisExpansion
    from linear import Linear

    def grad_update(step_size, variables, grads):
        for var, grad in zip(variables, grads):
            var.assign_sub(step_size * grad)

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    # num_inputs = 10
    num_outputs = 1
    num_basis = num_inputs

    x = rng.uniform(shape=(num_samples, 1))
    b = rng.normal(shape=(1, 1))
    y = rng.normal(
        shape=(num_samples, 1),
        mean=np.sin(2 * x * np.pi) + b,
        stddev=0.1,
    )

    bar = trange(20)

    linear = Linear(num_inputs, num_outputs)
    basis_module = BasisExpansion(rng, num_basis)
    init_loss = -1

    step_size = 0.1
    decay_rate = 0.999

    for i in bar:
        with tf.GradientTape() as tape:

            basis_fs = basis_module(x)
            y_hat = linear(basis_fs)
            if i == 0:
                init_loss = tf.math.reduce_mean((y - y_hat) ** 2)
            loss = tf.math.reduce_mean((y - y_hat) ** 2)

        grads = tape.gradient(loss, linear.trainable_variables+basis_module.trainable_variables)
        grad_update(step_size, linear.trainable_variables+basis_module.trainable_variables, grads)

        step_size *= decay_rate

    assert init_loss >= loss
