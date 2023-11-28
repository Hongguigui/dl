import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from tqdm import trange
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


class SineLinear(tf.Module):
    def __init__(self, num_inputs, num_outputs, omega_0=30, bias=True, is_first=False):
        rng = tf.random.get_global_generator()

        # stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))
        self.is_first = is_first

        self.omega_0 = omega_0
        if self.is_first:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval=-1/num_inputs, maxval=1/num_outputs),
                trainable=True,
                name="Linear/w",
            )
        else:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval=-np.sqrt(6 / num_inputs) / self.omega_0,
                            maxval=np.sqrt(6 / num_inputs) / self.omega_0), trainable=True, name="Linear/w",
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

    def __call__(self, x, act):
        z = x @ self.w

        if self.bias:
            z += self.b

        if act:
            # z = tf.math.sin(self.omega_0 * z)
            z = tf.math.cos(self.omega_0 * z)
            # z = tf.nn.relu(z)  # much slower and doesnt converge well
            # z = tf.nn.sigmoid(z)  # also doesnt work and get stuck at a fixed loss

        return z


class Siren(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, hidden_width):

        self.num_hidden = num_hidden
        self.layers = []

        input_layer = SineLinear(num_inputs, hidden_width, is_first=True)
        self.layers.append(input_layer)

        for i in range(num_hidden):
            hidden_layer = SineLinear(hidden_width, hidden_width)
            self.layers.append(hidden_layer)

        output_layer = SineLinear(hidden_width, num_outputs)
        self.layers.append(output_layer)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.y_hat = layer(x, act=True)
            # output no activation
            elif i == self.num_hidden+1:
                self.y_hat = layer(self.y_hat, act=None)
            else:
                self.y_hat = layer(self.y_hat, act=True)

        return self.y_hat


def rgb_to_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(img)
    # plt.imshow(img)
    # plt.show()
    m, n = img.shape
    R, C = np.mgrid[:m, :n]
    R = R / m
    C = C / n
    xy_coords = tf.convert_to_tensor(np.column_stack((C.ravel(), R.ravel())), dtype=tf.float32)
    pix_vals = tf.convert_to_tensor(img.ravel(), dtype=tf.float32)
    img = tf.convert_to_tensor(img)

    return img, xy_coords, pix_vals, [m, n]


def eval(model, grid_shape, path):
    x, y = grid_shape
    plt.figure(1)
    x_grid = np.linspace(0, 1, x, dtype="float32")
    y_grid = np.linspace(0, 1, y, dtype="float32")
    # (x, y) = np.meshgrid(x_grid, y_grid)
    z = np.zeros(grid_shape)
    print(z.shape)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            p = model(np.array([[y_grid[j], x_grid[i]]]))
            z[i, j] = p*255

    cv2.imwrite(path, z)
    # np.savetxt(path+'.txt', z)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment to use cpu
    gpus = tf.config.list_physical_devices('GPU')
    # need around 11 GB of ram to run the whole image as a batch ~ 90k samples
    # use cpu by default
    # if gpus:
    #     tf.config.set_logical_device_configuration(
    #         gpus[0],
    #         [tf.config.LogicalDeviceConfiguration(memory_limit=11000)]  # oof might crash stuff
    #     )
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    raw_image = './raw_image.jpg'
    image, coords, pixels, shape = rgb_to_gray(raw_image)

    # image, coords, pixels, shape = get_img_tensor(image)
    print(shape)
    # plt.plot(tf.reshape(pixels, shape).numpy())
    # plt.savefig(f'./iterations/output.jpg')
    # print(image)
    # tmp = tf.reshape(pixels, shape)
    # print(tmp)
    pixels = pixels/255

    data = np.column_stack((coords, pixels))
    print(data)
    print(coords.shape)
    print(pixels.shape)

    siren_model = Siren(2, 1, 3, 256)
    num_iters = 1500
    bar = trange(num_iters)
    refresh_rate = 1
    steps_til_summary = 50

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # 30k about 0.001 at 1500 epochs
    batch_size = 30000  # 30k > 45k > 90k large batch size trains way faster
    buf_size = 90000
    step_size = 0.01
    decay_rate = 0.999
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)  # seems to be better than sgd in this case
    train_dataset = tf.data.Dataset.from_tensor_slices((coords, pixels))
    train_dataset = train_dataset.shuffle(buf_size).batch(batch_size)

    regressor_total_parameters = 0
    for variable in siren_model.trainable_variables:
        var_shape = variable.get_shape()
        variable_parameters = 1
        for dim in var_shape:
            variable_parameters *= dim
        regressor_total_parameters += variable_parameters

    print("\nNumber of trainable parameters: ", regressor_total_parameters)

    for i in bar:
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                y_hat = siren_model(x_batch)
                y_hat = tf.squeeze(y_hat)
                loss = tf.math.reduce_mean(((y_hat - y_batch)**2))

            grads = tape.gradient(loss, siren_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, siren_model.trainable_variables))
            # grad_update(step_size, siren_model.trainable_variables, grads)
            # step_size *= decay_rate

            if step % refresh_rate == (refresh_rate - 1):
                bar.set_description(
                    f"Epoch {i} step {step}; Loss => {loss.numpy():0.5f}"
                )
                bar.refresh()

    eval(model=siren_model, grid_shape=[item//2 for item in shape], path='./output/output_under.png')
    eval(model=siren_model, grid_shape=shape, path='./output/output.png')
    # use more pixels than training data and see how the model does in the end
    # seems to work quite well surprisingly
    eval(model=siren_model, grid_shape=[item*5//4 for item in shape], path='./output/output_super.png')
    eval(model=siren_model, grid_shape=[item*2 for item in shape], path='./output/output_super_super.png')
