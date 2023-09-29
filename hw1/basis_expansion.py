import tensorflow as tf


# gaussian basis functions
class BasisExpansion(tf.Module):
    def __init__(self, rng, num_basis):
        self.num_basis = num_basis
        self.sigma = tf.Variable(rng.normal(shape=[self.num_basis]), trainable=True, name='Basis/sigma')
        self.mu = tf.Variable(rng.normal(shape=[self.num_basis]), trainable=True, name='Basis/mu')

    def __call__(self, x):
        self.basis = tf.math.exp(-1 * (((x - self.mu) ** 2) / (self.sigma ** 2)))
        return self.basis
