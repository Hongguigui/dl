import pytest


def test_trainable_variable_count():
    import tensorflow as tf

    from cifar import Classifier

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    kernels = [(3, 3)] * 2
    classifier = Classifier(input_depth=1, layer_depths=[2, 1], layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=(28, 28), num_channels=[4, 8])

    classifier_total_parameters = 0
    for variable in classifier.trainable_variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        classifier_total_parameters += variable_parameters

    assert classifier_total_parameters == 295962


@pytest.mark.parametrize("num_conv", [2, 4])
@pytest.mark.parametrize("num_fc", [2, 4])
@pytest.mark.parametrize("img_shape", [28, 32])
def test_classifier_init(num_conv, num_fc, img_shape):
    import tensorflow as tf

    from cifar import Classifier

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = rng.normal(shape=[1, img_shape, img_shape, 1])

    kernels = [(3, 3)] * 2
    classifier = Classifier(input_depth=1, layer_depths=[2, 1], layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=(img_shape, img_shape), num_channels=[4, 8])

    y_hat = classifier(x, dropout=0.1)

    tf.debugging.check_numerics(y_hat, message=str(y_hat))


@pytest.mark.parametrize("num_channels", [16, 32, 64])
def test_conv2d_shape(num_channels):
    import tensorflow as tf

    from cifar import Conv2d

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = rng.normal(shape=[1, 28, 28, 1])

    conv2d = Conv2d(num_channels, [3, 3], 1, 1, (28, 28))

    y_hat = conv2d(x)

    # conv2d.w_out_size = (((conv2d.W - conv2d.K1 + 2 * conv2d.P) / conv2d.S) + 1)
    # conv2d.h_out_size = (((conv2d.H - conv2d.K2 + 2 * conv2d.P) / conv2d.S) + 1)
    print(y_hat.shape)
    tf.assert_equal(y_hat.shape, (1, int(conv2d.h_out_size), int(conv2d.w_out_size), num_channels))
