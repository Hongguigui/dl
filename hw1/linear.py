# Hongyu Wu --- modified from example provided by Prof. Curro

import numpy as np
import tensorflow as tf

from basis_expansion import BasisExpansion


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
                # trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        # print(x.shape)
        # print(w.shape)
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config_noisy_sine.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_basis = 6

    num_samples = config["data"]["num_samples"]
    num_inputs = num_basis
    num_outputs = 1

    x = rng.uniform(shape=(num_samples, 1))
    w = rng.normal(shape=(1, 1))
    b = rng.normal(shape=(1, 1))
    y = rng.normal(
        shape=(num_samples, 1),
        mean=np.sin(2 * x * np.pi) + b,
        stddev=config["data"]["noise_stddev"],
    )

    # bases = BasisExpansion(rng=rng, num_basis=num_basis)

    linear = Linear(num_inputs, num_outputs)

    basis_module = BasisExpansion(rng, num_basis)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]
    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            basis_fs = basis_module(x_batch)

            y_hat = linear(basis_fs)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(loss, linear.trainable_variables+basis_module.trainable_variables)
        grad_update(step_size, linear.trainable_variables+basis_module.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots(1, 2)

    xs = np.linspace(0, 1, 100)[:, tf.newaxis]

    basis_xs = basis_module(xs)

    ax[0].plot(xs, linear(basis_xs), '--', color='red')

    ax[0].scatter(x, y, color="blue")
    ax[0].plot(xs, np.sin(2 * np.pi * xs) + tf.squeeze(b))

    # ax[0].plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")

    # a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    # ax.plot(a.numpy().squeeze(), linear(a).numpy().squeeze(), "-")

    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Sine fit using SGD")

    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    # plot the basis functions
    for mu, sig in zip(basis_module.mu.numpy(), basis_module.sigma.numpy()):
        ax[1].plot(
            xs, np.exp(-np.square(xs - mu) / (np.square(sig)))
        )

    ax[1].set_title("Gaussian basis functions")

    fig.savefig("./artifacts/plot.pdf")
