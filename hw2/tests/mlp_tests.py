import pytest


@pytest.mark.parametrize("num_hidden", [2, 4])
@pytest.mark.parametrize("hidden_width", [32, 64])
def test_mlp_init(num_hidden, hidden_width):
    import tensorflow as tf

    from mlp import MLP
    import numpy as np

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = np.asarray([(1, 1)])

    num_input = 1
    num_output = 1

    mlp_model = MLP(num_input, num_output, num_hidden, hidden_width)

    y_hat = mlp_model(x)

    tf.debugging.check_numerics(y_hat, message=str(y_hat))


# check if the nested trainable variables are accessible
# loop modified from
# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
@pytest.mark.parametrize("num_hidden", [1, 2])
@pytest.mark.parametrize("hidden_width", [4, 8])
def test_trainable_variable_count(num_hidden, hidden_width):
    import tensorflow as tf

    from linear import Linear
    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_input = 1
    num_output = 1

    mlp_model = MLP(num_input, num_output, num_hidden, hidden_width)

    mlp_total_parameters = 0
    for variable in mlp_model.trainable_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        # print(variable_parameters)
        mlp_total_parameters += variable_parameters

    linear_trainable = []
    input_layer = Linear(2, hidden_width)
    linear_trainable.extend([*input_layer.trainable_variables])
    for i in range(num_hidden):
        hidden_layer = Linear(hidden_width, hidden_width)
        linear_trainable.extend([*hidden_layer.trainable_variables])
    output_layer = Linear(hidden_width, 1)
    linear_trainable.extend([*output_layer.trainable_variables])

    linear_total_parameters = 0
    for variable in linear_trainable:
        shape = variable.shape
        # print(shape)
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        linear_total_parameters += variable_parameters

    # print('\nMLP model trainable variable count:', mlp_total_parameters)
    # print('\nSum of linear models trainable variable count:', linear_total_parameters)
    tf.assert_equal(mlp_total_parameters, linear_total_parameters)


# only passes when input are non-negative due to relu
def test_additivity():
    import tensorflow as tf

    from mlp import MLP
    import numpy as np

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 1
    num_outputs = 1
    num_hidden = 1
    hidden_width = 2

    mlp_model = MLP(num_inputs, num_outputs, num_hidden, hidden_width)

    a = np.asarray([(1., 1)])
    b = np.asarray([(2., 2.)])
    c = np.asarray([(3., 3.)])

    tf.debugging.assert_near(mlp_model(c), mlp_model(a) + mlp_model(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from mlp import MLP

    import numpy as np

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 1
    num_outputs = 1
    num_hidden = 1
    hidden_width = 2

    mlp_model = MLP(num_inputs, num_outputs, num_hidden, hidden_width)

    a = np.asarray([(1., 1.)])
    b = np.asarray([(2., 2.)])

    tf.debugging.assert_near(mlp_model(a * b), mlp_model(a) * b, summarize=2)


@pytest.mark.parametrize("num_samples", [1, 16, 128])
def test_dimensionality(num_samples):
    import tensorflow as tf

    from mlp import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 1
    num_outputs = 1
    num_hidden = 1
    hidden_width = 2

    mlp_model = MLP(num_inputs, num_outputs, num_hidden, hidden_width)

    a = [(1., 1.)]*num_samples
    z = mlp_model(a)

    print(z.shape)

    tf.assert_equal(tf.shape(z)[0], num_samples)
