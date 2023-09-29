import numpy as np
import pytest


# BasisExpansion class unit tests
@pytest.mark.parametrize("num_basis", [1, 4, 8, 16])
def test_basis_count(num_basis):
    import tensorflow as tf

    from linear import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = [1.]*num_basis

    base = BasisExpansion(rng, num_basis)
    basis_functions = base(x)
    tf.assert_equal(len(basis_functions), num_basis)


def test_non_linear():
    import tensorflow as tf

    from linear import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    basis_module = BasisExpansion(rng, num_inputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    assert ~np.isclose(basis_module(a + b), basis_module(a) + basis_module(b)).all()
    assert ~np.isclose(basis_module(a * b), basis_module(a) * b).all()
