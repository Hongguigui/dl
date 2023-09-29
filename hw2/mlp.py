import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay
from tqdm import trange

from linear import Linear


class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, hidden_width):

        self.layers = []

        input_layer = Linear(2, hidden_width)
        self.layers.append(input_layer)

        for i in range(num_hidden):
            hidden_layer = Linear(hidden_width, hidden_width)
            self.layers.append(hidden_layer)

        output_layer = Linear(hidden_width, 1)
        self.layers.append(output_layer)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.y_hat = layer(x)
            else:
                self.y_hat = layer(tf.nn.relu(self.y_hat))

        return self.y_hat
        # return self.y_hat


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def bce_loss(y, y_hat, weights):
    # smoothing = 1e-15
    alpha = 0.0001
    bce = tf.reduce_mean(-1 * y * tf.math.log(tf.nn.sigmoid(tf.squeeze(y_hat)) + 1e-15) - (1 - y) * tf.math.log(
        1 - tf.nn.sigmoid(tf.squeeze(y_hat)) + 1e-15))

    l2 = 0
    for weight in weights:
        l2 += tf.nn.l2_loss(weight.w)

    return tf.reduce_mean(bce + alpha * l2)
    # return tf.reduce_mean(bce)


def gen_spirals(num_samples=1000):
    # r = a*theta
    noise_std = 0.2
    rng = np.random.default_rng(12345)
    noise_0 = rng.normal(0, scale=noise_std, size=num_samples)
    rng = np.random.default_rng(54321)
    noise_1 = rng.normal(0, scale=noise_std, size=num_samples)

    # 1 - 15 so that center of spirals not touching
    r = np.linspace(1, 15, num_samples)

    t_0 = np.linspace(0, 15, num_samples)
    t_1 = np.linspace(0, 15, num_samples)

    x_0 = r * np.cos(t_0) + noise_0
    y_0 = r * np.sin(t_0) + noise_0
    x_1 = -1 * r * np.cos(t_1) - noise_1
    y_1 = -1 * r * np.sin(t_1) - noise_1

    x = np.array(np.concatenate([x_0, x_1]))
    y = np.array(np.concatenate([y_0, y_1]))
    xy = np.column_stack((x, y))
    xy = np.float32(xy)

    labels_0 = np.zeros(num_samples, dtype="float32")
    labels_1 = np.ones(num_samples, dtype="float32")
    labels = np.concatenate([labels_0, labels_1])

    return xy, labels


if __name__ == '__main__':

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_iters = 5000
    num_samples = 500
    step_size = 0.1
    decay_rate = 0.999
    batch_size = 64
    refresh_rate = 10

    mlp_model = MLP(1, 1, 8, 64)

    # generate the spirals data
    coords, class_labels = gen_spirals(num_samples)
    bar = trange(num_iters)

    for i in bar:
        batch_indices = tf_rng.uniform(
            shape=[batch_size], maxval=num_samples*2, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(coords, batch_indices)
            y_batch = tf.gather(class_labels, batch_indices)

            y_hat = mlp_model(x_batch)
            loss = bce_loss(y_batch, y_hat, mlp_model.layers)

        grads = tape.gradient(loss, mlp_model.trainable_variables)
        grad_update(step_size, mlp_model.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.5f}, step_size => {step_size:0.5f}"
            )
            bar.refresh()

    n = 301
    plt.figure(2)
    x_grid = np.linspace(-15, 15, n, dtype="float32")
    y_grid = np.linspace(-15, 15, n, dtype="float32")
    (x, y) = np.meshgrid(x_grid, y_grid)
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p = mlp_model(np.array([[x_grid[i], y_grid[j]]]))
            p = tf.nn.sigmoid(tf.squeeze(p))

            # decision boundary
            # if p >= 0.5:
            #     z[j, i] = 1

            # decision surface
            z[j, i] = p

    cm = plt.cm.RdBu
    display = DecisionBoundaryDisplay(xx0=x, xx1=y, response=z)
    display.plot()
    display.ax_.scatter(x=coords[:, 0], y=coords[:, 1], c=class_labels, alpha=1, edgecolors="black")
    plt.savefig('boundary.png')

'''
The greatest difficulty was actually fixing the loss function
Before fixing the loss always stuck at 0.7-ish and by inspecting the output
the model always outputs either 0 or 0.5 after applying sigmoid and the initial
suspicion was that the BCE loss was overwhelmed by the l2 penalty but decreasing
the hyperparameter did not improve loss and the model still could not learn

Applications with/without sigmoid at various places were also tried out before
finding out where the issue was and sigmoid is used to constraint the output range

It was found that y_hat had an additional axis that made the loss value erratic

The number of iteration of 5000 is about right since the loss is about 0.03
and the small value is likely due to the l2 penalty and cannot be reduced more

Width of 64 is chosen since the Linear module weight initialization is affected
by the input and output size hence larger input/output size might make weights too large

Depth was chosen to be 10 at the start to make sure that the model would not under fit
and later reduced by binary search the main consideration is the smoothness of the
decision boundary in addition to correctly classifying the spirals and depth of 8
is deemed to be the most optimal in terms of performance
'''
