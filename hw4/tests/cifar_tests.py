import pytest


# tests down sampling of a res block
@pytest.mark.parametrize("shape", [32, 64])
def test_res_blk_shape(shape):
    import tensorflow as tf

    from cifar import ResidualBlock

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    kernels = (3, 3)
    block = ResidualBlock(in_channels=32, out_channels=32, filter_size=kernels, input_size=(32, 32))

    image = tf.Variable(rng.normal(shape=[1, shape, shape, 32], stddev=0.1))

    y = block(image, 0)

    tf.assert_equal(tf.shape(y), (1, int(shape/2), int(shape/2), 32))


# check numerics for output
def test_classifier_init():
    import tensorflow as tf

    from cifar import Classifier

    img_shape = 32
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = rng.normal(shape=[1, img_shape, img_shape, 3])

    kernels = (3, 3)
    classifier = Classifier(input_depth=1,  layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=(img_shape, img_shape))

    y_hat = classifier(x, dropout=0)

    tf.debugging.check_numerics(y_hat, message=str(y_hat))


# check dropout for output
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_classifier_dropout(dropout):
    import numpy as np
    import tensorflow as tf

    from cifar import Classifier

    img_shape = 32
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x_0 = tf.ones(shape=[1, img_shape, img_shape, 3])
    x_1 = tf.ones(shape=[1, img_shape, img_shape, 3])

    kernels = (3, 3)
    classifier = Classifier(input_depth=1,  layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=(img_shape, img_shape))

    y_hat_0 = classifier(x_0, dropout=dropout)
    y_hat_1 = classifier(x_1, dropout=dropout)

    tf.assert_equal(x_0, x_1)

    if dropout:
        assert ~np.isclose(y_hat_0, y_hat_1).all()
    else:
        tf.assert_equal(y_hat_0, y_hat_1)


@pytest.mark.parametrize("img_shape", [32, 64])
def test_group_norm_shape(img_shape):
    import tensorflow as tf

    from cifar import GroupNorm

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    x = rng.normal(shape=[1, img_shape, img_shape, 32])
    gn = GroupNorm(32)
    y = gn(x)

    tf.assert_equal(tf.shape(y), tf.shape(x))
