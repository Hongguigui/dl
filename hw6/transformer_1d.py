import numpy as np
import tensorflow as tf
from tensorflow import (cast, convert_to_tensor, float32, int32, linalg,
                        matmul, maximum, ones, reshape, shape, transpose)

# import tensorflow_datasets as tfds // doesn't work


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/
def scaled_dot_product_attention(queries, keys, values, d_qkv, mask=None):
    scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(cast(d_qkv, float32))
    # softmax essentially ignores very negative numbers
    # mask = mask[:, tf.newaxis, :, :]

    if mask is not None:
        scores += -1e9 * tf.cast(mask, tf.float32)

    weights = tf.nn.softmax(scores)
    # print(0)
    # print(scores)
    # print(1)
    # print(weights)
    # print(1)
    # print(values)
    # print(1)
    # print(matmul(weights, values))

    return matmul(weights, values)


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


# include the embeddings
def get_param_count(model):
    total_parameters = 0
    for variable in model.trainable_variables:
        params_shape = variable.get_shape()
        variable_parameters = 1
        for dim in params_shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    print("\nNumber of trainable parameters: ", total_parameters)


class LayerNorm(tf.Module):
    def __init__(self, d_model, eps):
        # eps too small leads to loss exploding
        self.gamma = tf.Variable(tf.ones(shape=[d_model]), trainable=True)
        self.beta = tf.Variable(tf.zeros(shape=[d_model]), trainable=True)
        self.eps = eps

    def __call__(self, x):
        # N, L, E = x.get_shape().as_list()
        mean, var = tf.nn.moments(x, [1], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)

        return x * self.gamma + self.beta


# https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/
# https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/
# https://www.tensorflow.org/text/tutorials/transformer#setup
# https://github.com/karpathy/nanoGPT/blob/master/model.py
class MultiheadAttention(tf.Module):
    def __init__(self, h, d_qkv, d_model):
        assert d_model % h == 0
        self.heads = h  # Number of attention heads
        self.d_qkv = d_qkv  # Dimension of projected q k v

        self.w_q = Linear(d_model, d_qkv)  # weight for the queries
        self.w_k = Linear(d_model, d_qkv)  # weight for the keys
        self.w_v = Linear(d_model, d_qkv)  # weight for the values
        self.w_o = Linear(d_qkv, d_model)  # total dimension

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_qkv)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_qkv))
        return x

    # https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/
    def __call__(self, queries, keys, values, mask=None):
        # Rearrange to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.w_q(queries), self.heads, True)
        k_reshaped = self.reshape_tensor(self.w_k(keys), self.heads, True)
        v_reshaped = self.reshape_tensor(self.w_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        output_reshaped = scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped, self.d_qkv, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        # print(tf.shape(output_reshaped))
        output = self.reshape_tensor(output_reshaped, self.heads, False)
        # print(tf.shape(output))

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.w_o(output)


# tf tutorial example with keras
# only used in test cases to verify
class PositionalEmbedding_tutorial(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


# should be equivalent to the tf tutorial class
# the positional embeddings are fixed
class PositionalEmbedding(tf.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        rng = tf.random.get_global_generator()
        self.d_model = d_model
        self.embedding_w = tf.Variable(rng.normal(shape=[vocab_size, d_model]), trainable=True)
        self.pos_encoding = positional_encoding(length=seq_len, depth=d_model)

    # def compute_mask(self, *args, **kwargs):
    #     return self.embedding.compute_mask(*args, **kwargs)

    def __call__(self, x):
        length = tf.shape(x)[1]
        x = tf.nn.embedding_lookup(self.embedding_w, x)
        # This factor sets the relative scale of the embedding and positional_encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x


class MLP(tf.Module):
    def __init__(self, d_model):
        self.c_fc = Linear(d_model, d_model)
        self.c_proj = Linear(d_model, d_model)

    def __call__(self, x, dropout):
        x = self.c_fc(x)
        x = tf.nn.gelu(x)
        x = self.c_proj(x)
        x = tf.nn.dropout(x, dropout)

        return x


# architecture in the gpt paper
class TransformerBlock(tf.Module):
    def __init__(self, d_model, h):
        assert d_model % h == 0
        self.ln_1 = LayerNorm(d_model, eps=1e-5)
        self.attn = MultiheadAttention(h, d_model // h, d_model)
        self.ln_2 = LayerNorm(d_model, eps=1e-5)
        self.mlp = MLP(d_model)

    def __call__(self, x, dropout, mask):
        qkv = self.ln_1(x)
        # skip connections with scaling
        x = x + self.attn(qkv, qkv, qkv, mask)
        x = x + self.mlp(self.ln_2(x), dropout)

        return x


class TransformerDecoder(tf.Module):
    def __init__(self, seq_len, h, d_model, n, vocab_size):
        self.sublayers = [TransformerBlock(d_model, h) for _ in range(n)]
        self.embedding = PositionalEmbedding(vocab_size, d_model, seq_len)
        # num classes or vocab size
        num_classes = vocab_size
        self.output = Linear(d_model, num_classes)

    def causal_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def padding_mask(self, padding_input):
        # Create mask which marks the zero padding values in the input by a 1
        mask = tf.math.equal(padding_input, 0)
        mask = cast(mask, float32)

        return mask[:, tf.newaxis, tf.newaxis, :]

    def __call__(self, idx, dropout):
        padding_mask = self.padding_mask(idx)
        causal_mask = self.causal_mask(idx.shape[1])
        # # print(tf.shape(padding_mask))
        # # print(tf.shape(causal_mask))
        mask = maximum(padding_mask, causal_mask)
        # mask = get_causal_attention_mask(idx)

        x = tf.nn.dropout(self.embedding(idx), dropout)
        for layer in self.sublayers:
            x = layer(x, dropout, mask)
        x = self.output(x)

        return x


def scce_loss(y, y_hat, weights):
    alpha = 0.000001  # too much l2 overwhelms the softmax loss
    scce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    l2 = 0
    for weight in weights:
        # print(weight)
        l2 += tf.nn.l2_loss(weight)

    return tf.reduce_mean(scce + alpha * l2), alpha * l2


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def prepare_text(x, tokenizer, flag, seq_len):
    x = tokenizer.texts_to_sequences(x)
    if flag == 'train':
        y = [seq[1:seq_len + 1] for seq in x]
    elif flag == 'test':
        y = [seq[seq_len:seq_len + seq_len] for seq in x]
    else:
        return x, None
    x = [seq[:seq_len] for seq in x]

    print(y[0])
    print(x[0])

    # vocab_size = len(tokenizer.word_index) + 1
    x = pad_sequences(x, maxlen=seq_len, padding='post')
    y = pad_sequences(y, maxlen=seq_len, padding='post')
    # print(x_train[0])
    x = convert_to_tensor(x, dtype=int32)
    y = convert_to_tensor(y, dtype=int32)

    return x, y


def validate_model(iters, batch_size, x, y, model, mode):
    assert len(x) == len(y)
    validate_bar = trange(iters)
    val_refresh_rate = 10
    for k in validate_bar:
        batch_index = tf_rng.uniform(
            shape=[batch_size], maxval=len(x), dtype=tf.int32
        )
        x_val_batch = tf.gather(x, batch_index)
        y_val_batch = tf.gather(y, batch_index)

        y_val_hat = model(x_val_batch, 0)
        # tf.debugging.check_numerics(y_hat, message=str(y_hat))
        val_loss, _ = scce_loss(y_val_batch, y_val_hat, transformer.trainable_variables)
        # tf.debugging.check_numerics(loss, message=str(loss))

        if k % val_refresh_rate == 0:
            validate_bar.set_description(
                f"Step {k}; {mode}_loss => {val_loss.numpy():0.5f}"
            )
            validate_bar.refresh()


def demo_sequence_prediction(test_seq, test_seq_ans, model):
    test_text = tokenizer.sequences_to_texts(test_seq.numpy())
    # test_text_ans = tokenizer.sequences_to_texts(test_seq_ans.numpy())
    print(f'''\nOriginal sequence (processed): "{" ".join(test_text)}"''')
    # print(f'''Target sequence (processed): "{" ".join(test_text_ans)}"''')

    test_seq_padded = pad_sequences(test_seq, maxlen=seq_len, padding='post')
    pred = model(convert_to_tensor(test_seq_padded, dtype=int32), dropout=0)
    predicted = np.int64(tf.argmax(pred, 2))
    predicted_str = tokenizer.sequences_to_texts(predicted)
    # print(tf.shape(pred))
    # print(len(predicted_str))
    print(f'''\nPredicted sequence: "{" ".join(predicted_str)}"''')


if __name__ == '__main__':
    import pandas as pd
    from keras.preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from tqdm import trange

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf_rng = tf.random.get_global_generator()
    # tf_rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    # reusing ag_news files | probably not the best dataset for generation
    TRAIN_FILE_PATH = './train.csv'
    TEST_FILE_PATH = './test.csv'

    data = pd.read_csv(TRAIN_FILE_PATH)
    # data = data.head(10000)
    testdata = pd.read_csv(TEST_FILE_PATH)

    x_train = data['Description']  # Combine title and description (better accuracy than using them as separate features)
    # y_train = data['Title']  # Class labels need to begin from 0

    # x_train = x_train.head(1000)
    # y_train = y_train[:1000]

    x_test = testdata['Description']
    # y_test = testdata['Title']

    seq_len = 64
    vocab_size = 10000

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=42, shuffle=False)
    x_train, y_train = prepare_text(x_train, tokenizer, 'train', seq_len)
    x_val, y_val = prepare_text(x_val, tokenizer, 'train', seq_len)
    x_test, y_test = prepare_text(x_test, tokenizer, 'train', seq_len)

    # big n trains a bit too slowly per step
    transformer = TransformerDecoder(seq_len=seq_len, h=8, d_model=256, n=4, vocab_size=vocab_size)
    get_param_count(transformer)

    # training configs
    # much higher initial loss without scaling the residual connections and bigger model parameters
    num_iters = 6000
    num_samples = len(x_train)
    step_size = 0.01  # high learn rate also causes instability
    decay_rate = 0.9996  # not used when using optimizers
    batch_size = 32
    refresh_rate = 10
    validate_per_steps = num_iters // 4

    # threw type error with the embedding_lookup layer slices before
    # also does not seem to train the embeddings without the optimizer
    optimizer = tf.optimizers.SGD(learning_rate=step_size, momentum=0.9)

    bar = trange(num_iters)

    for i in bar:
        batch_indices = tf_rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x_train, batch_indices)
            y_batch = tf.gather(y_train, batch_indices)

            y_hat = transformer(x_batch, 0.05)
            tf.debugging.check_numerics(y_hat, message=str(y_hat))
            # final loss should be around 3-5 with 6000 steps on the whole dataset
            loss, l2_loss = scce_loss(y_batch, y_hat, transformer.trainable_variables)
            tf.debugging.check_numerics(loss, message=str(loss))

        grads = tape.gradient(loss, transformer.trainable_variables)
        # grad_update(step_size, transformer.trainable_variables, grads)
        optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

        step_size *= decay_rate
        optimizer.lr.assign(step_size)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.5f}, step_size => {step_size:0.6f}, l2_penalty => {l2_loss:0.5f},"
            )
            bar.refresh()

        if i % validate_per_steps == (validate_per_steps - 1):
            # optimizer = tf.optimizers.SGD(learning_rate=step_size, momentum=0.9)
            validate_model(iters=1, batch_size=batch_size, x=x_val, y=y_val, model=transformer, mode='val')

    validate_model(iters=10, batch_size=batch_size, x=x_test, y=y_test, model=transformer, mode='test')

    # demo after training | using the printed sequence from the training set
    x_train_df = data['Description'].head(1)

    test_seq_train, test_seq_ans_train = prepare_text(x_train_df, tokenizer, 'test', seq_len)
    # using printed sequence from the test set
    x_test_df = testdata['Description'].head(1)

    test_seq_test, test_seq_ans_test = prepare_text(x_test_df, tokenizer, 'test', seq_len)

    demo_sequence_prediction(test_seq_train, test_seq_ans_train, transformer)
    print('\nOriginal sequence: ', x_train_df.iloc[0])

    demo_sequence_prediction(test_seq_test, test_seq_ans_test, transformer)
    print('\nOriginal sequence: ', x_test_df.iloc[0])
    # still quite gibberish
    '''
    Original sequence (processed): "reuters short sellers wall street's band of ultra are seeing green again"
    Predicted sequence: "the to to street is in the in the in bay to"    
    Original sequence:  Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.
    
    Original sequence (processed): "unions representing workers at turner say they are after talks with stricken parent firm federal mogul"    
    Predicted sequence: "and of in the in they have the a with the to company the reserve of"    
    Original sequence:  Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.
    '''
