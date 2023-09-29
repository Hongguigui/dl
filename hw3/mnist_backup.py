import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange
from sklearn.model_selection import train_test_split
from linear import Linear
import gzip


class Conv2d(tf.Module):
    def __init__(self, filter_count, filter_size, strides, input_depth, input_size):
        rng = tf.random.get_global_generator()
        self.filter_count = filter_count
        self.filter_size = filter_size
        self.strides = strides
        shape = [filter_size[0], filter_size[1], input_depth, filter_count]
        self.w = tf.Variable(
            rng.normal(shape=shape),
            trainable=True)
        self.input_dim = input_size

        self.b = tf.Variable(tf.zeros(
                    shape=[1, filter_count],
                ),
                trainable=True)

        # Here W = Input size
        #         K = Filter size
        #         S = Stride
        #         P = Padding
        self.W = self.input_dim[1]
        self.H = self.input_dim[0]
        self.K1 = filter_size[0]
        self.K2 = filter_size[1]
        self.P = 0
        self.S = 1

        self.w_out_size = (((self.W - self.K1 + 2*self.P) / self.S) + 1)
        self.h_out_size = (((self.H - self.K2 + 2*self.P) / self.S) + 1)

    def __call__(self, x):
        self.y_hat = tf.nn.conv2d(x, self.w, [self.strides]*4, padding='VALID')
        return tf.nn.relu(self.y_hat + self.b)


class Classifier(tf.Module):
    def __init__(self, input_depth, layer_depths, layer_kernel_sizes, num_classes, hidden_width, input_dim, num_channels):
        # assume that all conv2d layers use the same filters
        fc_width = 64
        fc_count = 2
        rng = tf.random.get_global_generator()
        self.channels = input_depth
        self.conv_count = layer_depths[0]
        self.fc_count = layer_depths[1]
        self.kernels = layer_kernel_sizes
        self.output_size = num_classes
        self.input_dim = input_dim
        self.input_depth = input_depth

        self.conv_layers = []
        self.fc_layers = []
        self.num_channels = num_channels
        self.hidden_width = hidden_width
        self.num_classes = num_classes
        self.current_dim = [0, 0]
        self.dropout = 0

        for i in range(self.conv_count):
            if i == 0:
                self.conv_layers.append(Conv2d(self.num_channels[i], self.kernels[i], 1, self.input_depth, self.input_dim))
                self.current_dim = [self.conv_layers[-1].h_out_size, self.conv_layers[-1].w_out_size]
            else:
                self.conv_layers.append(Conv2d(self.num_channels[i], self.kernels[i], 1, self.num_channels[i-1], self.current_dim))
                self.current_dim = [self.conv_layers[-1].h_out_size, self.conv_layers[-1].w_out_size]

        fc_input = self.current_dim[0] * self.current_dim[1] * self.num_channels[-1]

        for i in range(self.fc_count):
            if i == 0:
                self.fc_layers.append(Linear(int(fc_input), self.hidden_width))
            else:
                self.fc_layers.append(Linear(self.hidden_width, self.hidden_width))

        if self.fc_count == 0:
            self.fc_layers.append(Linear(int(fc_input), self.num_classes))
        else:
            self.fc_layers.append(Linear(self.hidden_width, self.num_classes))

    def __call__(self, x, dropout):
        self.dropout = dropout
        for i, layer in enumerate(self.conv_layers):
            if i == 0:
                self.y_hat = layer(x)
            else:
                # self.y_hat = tf.nn.dropout(self.y_hat, rate=self.dropout, seed=23456)
                self.y_hat = layer(self.y_hat)
        self.y_hat = flatten(self.y_hat)
        for layer in self.fc_layers:
            self.y_hat = tf.nn.dropout(self.y_hat, rate=self.dropout, seed=12345)
            self.y_hat = layer(tf.nn.relu(self.y_hat))

        return self.y_hat


def flatten(X):
    X_shape = tf.shape(X)
    batch_size = X_shape[0]
    new_shape = [batch_size, tf.math.reduce_prod(X_shape[1:])]
    return tf.reshape(X, new_shape)


def random_plots(x, y, name):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        label_index = int(y[i])
        plt.title(str(label_index))
    plt.savefig(f"./images/{name}_digit_samples.png")
    plt.close()


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def scce_loss(y, y_hat, weights):
    alpha = 0.0001
    scce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    l2 = 0
    for weight in weights:
        l2 += tf.nn.l2_loss(weight.w)

    return tf.reduce_mean(scce + alpha * l2), alpha * l2


def evaluate_model(x, y, model, name):
    eval_y_hat = classifier(x, dropout=0)
    eval_correct_prediction = tf.equal(tf.argmax(eval_y_hat, 1), y)
    eval_accuracy = tf.reduce_mean(tf.cast(eval_correct_prediction, tf.float32))
    eval_loss, l2_loss = scce_loss(y, eval_y_hat, model.fc_layers+model.conv_layers)
    print(f"{name}_loss: ", eval_loss.numpy())
    print(f"{name}_acc: ", eval_accuracy.numpy())


# skips 16
def get_x(x_file, image_dims):
    height = image_dims[0]
    width = image_dims[1]
    x_train = gzip.open(x_file, 'rb')
    x_train.read(16)
    x_train = np.frombuffer(x_train.read(), dtype=np.uint8).astype(np.float32).reshape(-1, height, width, 1)
    x_train = x_train / 255

    return x_train


# skips 8
def get_y(y_file):
    y_data = gzip.open(y_file, 'rb')
    y_data.read(8)
    y_data = np.frombuffer(y_data.read(), dtype=np.uint8).astype(np.float32)

    return y_data



if __name__ == '__main__':

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    image_rows = 28
    image_cols = 28
    batch_size = 128
    image_shape = (image_rows, image_cols, 1)
    image_dims = [image_rows, image_cols]

    x_train = get_x('train-images-idx3-ubyte.gz', image_dims)
    y_train = get_y('train-labels-idx1-ubyte.gz')
    x_test = get_x('t10k-images-idx3-ubyte.gz', image_dims)
    y_test = get_y('t10k-labels-idx1-ubyte.gz')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    y_train = np.int32(y_train)
    y_val = np.int32(y_val)
    y_test = np.int32(y_test)

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    num_iters = 3000
    step_size = 0.1
    decay_rate = 0.999
    refresh_rate = 10

    # x_train = x_train.reshape(x_train.shape[0], *image_shape)
    # x_val = x_val.reshape(x_val.shape[0], *image_shape)
    # x_test = x_test.reshape(x_test.shape[0], *image_shape)

    input_shape = x_train[0].shape
    random_plots(x_train, y_train, 'train')
    random_plots(x_val, y_val, 'val')
    random_plots(x_test, y_test, 'test')
    kernels = [(3, 3)] * 2
    # layer_depths, layer_kernel_sizes, num_classes, hidden_width, input_dim, num_filters
    classifier = Classifier(input_depth=x_train.shape[-1], layer_depths=[2, 1], layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=x_train[0].shape, num_channels=[8, 16])

    classifier_total_parameters = 0
    for variable in classifier.trainable_variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        classifier_total_parameters += variable_parameters

    print("\nNumber of trainable parameters: ", classifier_total_parameters)

    bar = trange(num_iters)
    train_loss = np.zeros(num_iters)
    train_acc = np.zeros(num_iters)

    for i in bar:
        batch_indices = tf_rng.uniform(
            shape=[batch_size], maxval=len(x_train), dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x_train, batch_indices)
            y_batch = tf.gather(y_train, batch_indices)

            y_hat = classifier(x_batch, dropout=0.1)

            loss, penalty = scce_loss(y_batch, y_hat, classifier.fc_layers+classifier.conv_layers)
            train_loss[i] = loss
            correct_prediction_train = tf.equal(np.int64(tf.argmax(y_hat, 1)), y_batch)
            train_accuracy = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))

        grads = tape.gradient(loss, classifier.trainable_variables)
        grad_update(step_size, classifier.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.5f}, step_size => {step_size:0.5f}, accuracy => {train_accuracy:0.5f}, l2_penalty => {penalty:0.5f}"
            )
            bar.refresh()

    evaluate_model(x_val, y_val, classifier, 'validation')
    evaluate_model(x_test, y_test, classifier, 'test')
