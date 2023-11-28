import pytest


# tests shape of the embedding layers
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("sequence_len", [32, 64])
def test_pos_encoding_shape(batch_size, sequence_len):
    import tensorflow as tf

    from transformer_1d import (PositionalEmbedding,
                                PositionalEmbedding_tutorial)

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    sequences = rng.uniform(shape=[batch_size, sequence_len], minval=1, maxval=1000, dtype=tf.dtypes.int32)

    vocab_size = 1000
    d_model = 512
    embed_impl = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, seq_len=sequence_len)
    embed_tutor = PositionalEmbedding_tutorial(vocab_size=vocab_size, d_model=d_model)
    embedded_i = embed_impl(sequences)
    embedded_t = embed_tutor(sequences)
    # Check the shape
    tf.assert_equal(tf.shape(embedded_t), tf.shape(embedded_i))


def test_multihead_attention_shape():
    import tensorflow as tf
    from numpy import random

    from transformer_1d import MultiheadAttention

    h = 8  # Number of self-attention heads
    d_qkv = 64
    d_model = 512  # Dimensionality of the model sub-layers outputs
    batch_size = 64  # Batch size from the training process

    input_seq_length = 16  # Maximum length of the input sequence

    # embedded input with dimension d_model
    queries = random.random((batch_size, input_seq_length, d_model))
    keys = random.random((batch_size, input_seq_length, d_model))
    values = random.random((batch_size, input_seq_length, d_model))

    multihead_attention = MultiheadAttention(h, d_qkv, d_model)
    tf.assert_equal(tf.shape(multihead_attention(queries, keys, values)).numpy(), [batch_size, input_seq_length, d_model])


def test_layer_norm():
    import numpy as np
    import tensorflow as tf
    from keras.layers import LayerNormalization
    from numpy import random

    from transformer_1d import LayerNorm

    d_model = 512  # Dimensionality of the model sub-layer outputs
    batch_size = 64  # Batch size from the training process

    input_seq_length = 5  # Maximum length of the input sequence

    embedding = random.random((batch_size, input_seq_length, d_model))
    embedding = np.float32(embedding)

    # compare with tf implementation
    ln = LayerNorm(d_model, eps=1e-8)
    ln_keras = LayerNormalization(epsilon=1e-8, axis=1)
    ln_output = ln(embedding)
    keras_output = ln_keras(embedding)

    tf.assert_equal(tf.shape(ln_output), tf.shape(keras_output))
    # close enough that implementation is probably correct
    assert np.isclose(ln_output.numpy(), keras_output.numpy(), rtol=1e-5, atol=1e-5).all()
    # tf.debugging.assert_near(ln_output, keras_output, summarize=2)


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("h", [4, 8])
@pytest.mark.parametrize("d_model", [256, 512])
def test_transformer_init(seq_len, n, h, d_model):
    import tensorflow as tf

    from transformer_1d import TransformerDecoder

    batch_size = 64
    test_x = tf.random.uniform(shape=[batch_size, seq_len], dtype=tf.int32, maxval=100)

    transformer = TransformerDecoder(seq_len=seq_len, h=h, d_model=d_model, n=n, vocab_size=101)
    y = transformer(test_x, 0.)
    tf.debugging.check_numerics(y, message=str(y))


def test_causal_mask():
    import tensorflow as tf
    # from transformer_1d import scce_loss
    from transformer_1d import TransformerDecoder

    def scce_loss(y, y_hat):
        # too much l2 overwhelms the softmax loss
        scce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)

        return scce

    seq_len = 4
    # test_seq_0 = tf.constant(tf.random.uniform(shape=[1, seq_len], dtype=tf.int32, maxval=1000))
    x_0 = tf.Variable([[1, 1, 2, 3]])
    y_0 = tf.Variable([[1, 2, 3, 3]])  # shifted
    # print(x_0)

    transformer = TransformerDecoder(seq_len=seq_len, h=1, d_model=8, n=2, vocab_size=4)

    with tf.GradientTape() as tape:
        tape.watch(x_0)
        tape.watch(y_0)
        y_hat = transformer(x_0, 0.)
        loss = scce_loss(y_0, y_hat)

    # print(tf.nn.embedding_lookup(transformer.embedding.embedding_w, x_0))
    # grads = tape.gradient(loss, tf.nn.embedding_lookup(transformer.embedding.embedding_w, x_0))  # None
    # print(loss[0])
    # print(tf.gather(loss, 0))

    # seems to need jacobian to get the detailed derivatives
    grads = tape.jacobian(target=loss, sources=transformer.embedding.embedding_w)
    grads = tf.convert_to_tensor(grads)
    print(grads)
    # seems to leak information but also can't realize what is wrong
    # checked the matmuls in the attention function and seem to work as intended
    # tf.assert_equal(tf.argmax(y_0[0], 1)[1], tf.argmax(y_0[1], 1)[1])


# autoregressive text generating interface
def test_autoregressive_generation():
    import numpy as np
    import tensorflow as tf
    from tensorflow import convert_to_tensor

    from transformer_1d import TransformerDecoder

    seq_len = 16
    new_max_tokens = 32
    start_seq = [[12, 3, 8, 16, 25]]
    seq = convert_to_tensor(start_seq)
    transformer = TransformerDecoder(seq_len=seq_len, h=2, d_model=8, n=1, vocab_size=100)
    while tf.shape(seq)[1] < new_max_tokens:
        print('\nCurrent sequence: ', seq)
        print(f'\nUsing {seq[:, -seq_len:].numpy()} as input')
        new_token = tf.constant([np.int32(tf.argmax(transformer(seq[:, -seq_len:], 0.0)[-1], 1)[-1])])
        new_token = new_token[:, tf.newaxis]
        seq = tf.concat([seq, new_token], axis=1)
        seq = convert_to_tensor(seq)

    print('\nFinal sequence: ', seq)
