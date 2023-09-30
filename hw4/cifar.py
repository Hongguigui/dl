import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import trange
from tensorflow.python.client import device_lib
import matplotlib
matplotlib.use('TkAgg')
import time


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


def load_pickle(f):
    return pickle.load(f, encoding='latin1')


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data():
    # Load the raw CIFAR-10 data
    cifar10_dir = './cifar-10-batches-py/'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_100():
    data_pre_path = './cifar-100-python/'  # change this path
    # File paths
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    # Read dictionary
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    # Get data (change the coarse_labels if you want to use the 100 classes)
    data_train = data_train_dict[b'data']
    label_train = np.array(data_train_dict[b'fine_labels'])
    data_test = data_test_dict[b'data']
    label_test = np.array(data_test_dict[b'fine_labels'])
    data_train = np.float32(data_train / 255)
    data_test = np.float32(data_test / 255)

    return data_train, label_train, data_test, label_test


# adapted from https://github.com/taki0112/Group_Normalization-Tensorflow
# and https://amaarora.github.io/posts/2020-08-09-groupnorm.html
# and https://arxiv.org/abs/1803.08494
# normalize along axes H W C//G
class GroupNorm(tf.Module):
    def __init__(self, num_channels):
        self.gamma = tf.Variable(tf.ones(shape=[1, 1, 1, num_channels]), trainable=True)
        self.beta = tf.Variable(tf.zeros(shape=[1, 1, 1, num_channels]), trainable=True)
        self.G = 32

    def __call__(self, x):
        N, H, W, C = x.get_shape().as_list()
        self.G = min(self.G, C)
        x = tf.reshape(x, [N, H, W, self.G, C // self.G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)
        x = tf.reshape(x, [N, H, W, C])
        # print(x.shape)

        return x * self.gamma + self.beta


def gn_relu(gn, x):
    x_gn = gn(x)
    gr = tf.nn.relu(x_gn)

    return gr

# identity and convolutional blocks
class ResidualBlock(tf.Module):
    def __init__(self, in_channels, out_channels, filter_size, input_size):
        # conv block
        (self.h, self.w) = input_size
        self.h = int(self.h / 2)
        self.w = int(self.w / 2)
        new_dim = (self.h, self.w)

        # id block
        self.conv3 = Conv2d(in_channels, filter_size, 1, in_channels, input_size)
        self.conv4 = Conv2d(in_channels, filter_size, 1, in_channels, input_size)

        self.gn3 = GroupNorm(in_channels)
        self.gn4 = GroupNorm(in_channels)

        self.gn_id_0 = []
        self.gn_id_0.extend([self.gn3, self.gn4])

        self.conv_id_0 = []
        self.conv_id_0.extend([self.conv3, self.conv4])

        self.conv5 = Conv2d(in_channels, filter_size, 1, in_channels, input_size)
        self.conv6 = Conv2d(in_channels, filter_size, 1, in_channels, input_size)

        self.gn5 = GroupNorm(in_channels)
        self.gn6 = GroupNorm(in_channels)

        self.gn_id_1 = []
        self.gn_id_1.extend([self.gn5, self.gn6])

        self.conv_id_1 = []
        self.conv_id_1.extend([self.conv5, self.conv6])

        # conv block
        self.conv0 = Conv2d(out_channels, filter_size, 2, input_depth=in_channels, input_size=input_size)
        self.conv1 = Conv2d(out_channels, filter_size, 1, out_channels, new_dim)

        self.conv_blk_conv = []
        self.conv_blk_conv.extend([self.conv0, self.conv1])

        self.gn0 = GroupNorm(in_channels)
        self.gn1 = GroupNorm(out_channels)
        self.gn2 = GroupNorm(out_channels)

        self.conv_blk_gn = []
        self.conv_blk_gn.extend([self.gn0, self.gn1])

        # 1x1 layer
        self.conv2 = Conv2d(out_channels, (1, 1), 2, in_channels, input_size)

    def conv_block(self, blk_input, dropout):
        skip = self.conv2(blk_input)
        y = None
        for i, (gn, conv) in enumerate(zip(self.conv_blk_gn, self.conv_blk_conv)):
            if i == 0:
                y = gn_relu(gn, blk_input)
                y = conv(y)

            else:
                y = gn_relu(gn, y)
                if dropout:
                    y = tf.nn.dropout(y, rate=0.1)
                y = conv(y)

        # y = self.gn2(y)
        y = tf.math.add(y, skip)
        # y = tf.nn.relu(y)

        return y

    def id_block(self, id_input, gn_list, conv_list, dropout):
        # identity = id_input
        # y_id = gn_relu(self.gn3, id_input)
        # y_id = self.conv3(y_id)
        # y_id = gn_relu(self.gn4, y_id)
        # if dropout:
        #     y_id = tf.nn.dropout(y_id, rate=0.1)
        # y_id = self.conv4(y_id)
        # y_id = tf.math.add(y_id, identity)

        identity = id_input
        y_id = gn_relu(gn_list[0], id_input)
        y_id = conv_list[0](y_id)
        y_id = gn_relu(gn_list[1], y_id)
        if dropout:
            y_id = tf.nn.dropout(y_id, rate=0.1)
        y_id = conv_list[1](y_id)
        y_id = tf.math.add(y_id, identity)

        return y_id

    def stack_id_block(self, stack_input, dropout):
        y_id_0 = self.id_block(stack_input, self.gn_id_0, self.conv_id_0, dropout)
        if dropout:
            y_id_0 = tf.nn.dropout(y_id_0, rate=0.1)
        y_id_1 = self.id_block(y_id_0, self.gn_id_1, self.conv_id_1, dropout)

        return y_id_1

    def __call__(self, x, dropout):
        y_hat = self.stack_id_block(x, dropout)
        if dropout:
            y_hat = tf.nn.dropout(y_hat, rate=0.1)
        # y_hat = self.id_block(y_hat, dropout)
        y_hat = self.conv_block(y_hat, dropout)

        return tf.nn.relu(y_hat)


class Conv2d(tf.Module):
    def __init__(self, filter_count, filter_size, strides, input_depth, input_size):
        rng = tf.random.get_global_generator()
        self.filter_count = filter_count
        self.filter_size = filter_size
        self.strides = strides
        shape = [filter_size[0], filter_size[1], input_depth, filter_count]
        stddev = np.float32(tf.math.sqrt(2 / np.prod(shape)))
        self.w = tf.Variable(
            rng.normal(shape=shape, stddev=stddev),
            trainable=True)
        self.input_dim = input_size

        self.b = tf.Variable(tf.zeros(
                    shape=[1, filter_count],
                ),
                trainable=True)

    def __call__(self, x):
        self.y_hat = tf.nn.conv2d(x, self.w, [1] + [self.strides]*2 + [1], padding='SAME')
        return self.y_hat + self.b


class Classifier(tf.Module):
    def __init__(self, input_depth, layer_kernel_sizes, num_classes, hidden_width, input_dim):
        self.channels = input_depth
        self.kernels = layer_kernel_sizes
        self.output_size = num_classes
        self.input_dim = input_dim
        self.input_depth = input_depth

        self.conv_layers = []
        self.blocks = []
        self.fc_layers = []
        self.hidden_width = hidden_width
        self.num_classes = num_classes
        self.current_dim = self.input_dim
        self.dropout = 0

        self.conv0 = Conv2d(32, self.kernels, 1, 3, self.input_dim)
        self.blocks.append(
            ResidualBlock(in_channels=32, out_channels=64, filter_size=self.kernels, input_size=(32, 32)))
        self.blocks.append(
            ResidualBlock(in_channels=64, out_channels=64, filter_size=self.kernels, input_size=(16, 16)))
        self.blocks.append(
            ResidualBlock(in_channels=64, out_channels=128, filter_size=self.kernels, input_size=(8, 8)))
        # self.blocks.append(
        #     ResidualBlock(in_channels=128, out_channels=128, filter_size=self.kernels, input_size=(4, 4)))
        # self.blocks.append(
        #     ResidualBlock(in_channels=64, out_channels=128, filter_size=self.kernels, input_size=(2, 2)))

        # self.fc0 = Linear(num_inputs=256, num_outputs=256)
        self.fc1 = Linear(num_inputs=512, num_outputs=self.num_classes)

    def __call__(self, x, dropout):
        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0
        self.y_hat = tf.nn.relu(self.conv0(x))
        for block in self.blocks:
            if dropout:
                self.y_hat = tf.nn.dropout(self.y_hat, rate=self.dropout)
                self.y_hat = block(self.y_hat, 1)
                # self.dropout += 0.05
            else:
                self.y_hat = block(self.y_hat, 0)

        self.y_hat = tf.nn.avg_pool(self.y_hat, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        if dropout:
            self.dropout = 0.5
        self.y_hat = flatten(self.y_hat)
        # self.y_hat = tf.nn.relu(self.fc0(self.y_hat))
        if dropout:
            self.y_hat = tf.nn.dropout(self.y_hat, rate=self.dropout)
        self.y_hat = self.fc1(self.y_hat)

        return self.y_hat


def flatten(X):
    X_shape = tf.shape(X)
    batch_size = X_shape[0]
    new_shape = [batch_size, tf.math.reduce_prod(X_shape[1:])]
    return tf.reshape(X, new_shape)


def random_plots(x, y, name, labels):
    x = x[:25]
    # z = img_aug(x, None)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        label_index = int(y[i])
        plt.title(str(labels[label_index]))
    plt.savefig(f"./images/{name}_samples.png")
    plt.close()

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(z[i])
    #     label_index = int(y[i])
    #     plt.title(str(labels[label_index]))
    # plt.savefig(f"./images/augmented_{name}_samples.png")
    # plt.close()


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


# initial loss should be about -ln(.1) = 2.3 without l2
def scce_loss(y, y_hat, weights):
    alpha = 0.00001
    scce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    l2 = 0
    for weight in weights:
        # print(weight)
        l2 += tf.nn.l2_loss(weight)

    return tf.reduce_mean(scce + alpha * l2), alpha * l2


def evaluate_model(model, dataset, name, epochs):
    acc_list = []
    top_k_acc_list = []
    for step, (x_batch, y_batch) in enumerate(dataset):
        y_hat = model(x_batch, dropout=None)
        correct_prediction = tf.equal(np.int64(tf.argmax(y_hat, 1)), y_batch)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_list.append(accuracy)
        top_k_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_batch, y_hat, k=5), tf.float32))
        top_k_acc_list.append(top_k_acc)

    check_acc = tf.add_n(acc_list) / len(acc_list)
    top_k_acc = tf.add_n(top_k_acc_list) / len(acc_list)

    if epochs:
        print(f'\nepoch_{epochs} {name}_acc: ', check_acc.numpy())
        print(f'epoch_{epochs} {name}_top_5_acc: ', top_k_acc.numpy())
    else:
        print(f'\n{name}_acc: ', check_acc.numpy())
        print(f'{name}_top_5_acc: ', top_k_acc.numpy())

    return check_acc, top_k_acc


def img_aug(img_batch, seed):
    images, label = img_batch
    result = tf.image.random_flip_left_right(images)
    result = tf.image.random_brightness(result, 0.2)
    result = tf.image.random_hue(result, max_delta=0.2)
    result = tf.image.random_contrast(result, lower=0.6, upper=1.2)
    result = tf.clip_by_value(result, clip_value_min=0., clip_value_max=1.)

    return result, label


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print(device_lib.list_local_devices())

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    image_rows = 32
    image_cols = 32
    image_shape = (image_rows, image_cols, 3)
    image_dims = [image_rows, image_cols]
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cifar = 10
    if cifar == 10:
        x_train, y_train, x_test, y_test = get_CIFAR10_data()
    else:
        x_train, y_train, x_test, y_test = get_100()

    # print('Train data shape: ', x_train.shape)
    # print('Train labels shape: ', y_train.shape)
    # print('Test data shape: ', x_test.shape)
    # print('Test labels shape: ', y_test.shape)

    x_train = x_train.reshape((len(x_train), 3, 32, 32)).transpose(0, 2, 3, 1)
    x_test = x_test.reshape((len(x_test), 3, 32, 32)).transpose(0, 2, 3, 1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    y_train = np.int32(y_train)
    y_val = np.int32(y_val)
    y_test = np.int32(y_test)

    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    num_epochs = 150
    step_size = 0.01
    decay_rate = 0.98
    refresh_rate = 10
    batch_size = 64

    input_shape = x_train[0].shape
    # random_plots(x_train, y_train, 'train', classes)
    # random_plots(x_val, y_val, 'val', classes)
    # random_plots(x_test, y_test, 'test', classes)

    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    buf_size = 1000

    plain_train_dataset = train_dataset
    plain_train_dataset = plain_train_dataset.shuffle(buf_size).batch(batch_size)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    counter = tf.data.experimental.Counter()
    train_dataset = tf.data.Dataset.zip((train_dataset, (counter, counter)))
    train_dataset = train_dataset.map(img_aug, num_parallel_calls=AUTOTUNE).shuffle(buf_size).batch(
        batch_size).prefetch(AUTOTUNE)

    val_dataset = val_dataset.batch(64)
    test_dataset = test_dataset.batch(64)

    fine_tune = 0
    validate_per_epochs = 10

    path = f'./checkpoint/'
    kernels = (3, 3)
    classifier = Classifier(input_depth=x_train.shape[-1], layer_kernel_sizes=kernels,
                            num_classes=10, hidden_width=64, input_dim=x_train[0].shape)

    ckpt = tf.train.Checkpoint(classifier=classifier)
    manager = tf.train.CheckpointManager(
        ckpt, directory="./checkpoint/", max_to_keep=5)

    if fine_tune == 1:
        status = ckpt.restore(manager.latest_checkpoint)
        step_size = 0.006  # warm restart with higher lr
        decay_rate = 0.96  # maybe faster decay
        validate_per_epochs = 2
        # num_epochs = 100
        status.assert_consumed()



    classifier_total_parameters = 0
    for variable in classifier.trainable_variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        classifier_total_parameters += variable_parameters

    print("\nNumber of trainable parameters: ", classifier_total_parameters)

    bar = trange(num_epochs)

    for i in bar:
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # abandons tf.gather due to index validation slowing down
            with tf.GradientTape() as tape:
                # random_plots(x_batch, y_batch, 'training', classes)  # check if augmentation is working
                # t0 = time.time()
                y_hat = classifier(x_batch, dropout=0.1)
                # t1 = time.time()
                loss, penalty = scce_loss(y_batch, y_hat, classifier.trainable_variables)
                train_loss[i] = loss
                correct_prediction_train = tf.equal(np.int64(tf.argmax(y_hat, 1)), y_batch)
                train_accuracy = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))
                top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_batch, y_hat, k=5), tf.float32))

            # get the final layer as trainable_variables
            # train_vars = tf.Graph.get_collection(classifier.TRAINABLE_VARIABLES, name="Linear/w:0")
            # train_vars += tf.Graph.get_collection(classifier.TRAINABLE_VARIABLES, name="Linear/b:0")

            grads = tape.gradient(loss, classifier.trainable_variables)
            grad_update(step_size, classifier.trainable_variables, grads)

            if step % refresh_rate == (refresh_rate - 1):
                bar.set_description(
                    f"Epoch {i} Step {step}; Loss => {loss.numpy():0.5f}, step_size => {step_size:0.5f}, accuracy => {train_accuracy:0.5f},"
                    f" l2_penalty => {penalty:0.5f}, top_5_acc => {top5_acc:0.5f}"
                )
                bar.refresh()

        step_size *= decay_rate
        if i >= 0.8*num_epochs:  # lower learn rate even more at the end
            step_size *= decay_rate
            validate_per_epochs = 5
        if ((i % validate_per_epochs == validate_per_epochs-1) & (i > 0)) or (i % validate_per_epochs == 3 & i >= 0.7*num_epochs):
            val_acc, val_top_k_acc = evaluate_model(classifier, val_dataset, 'validation', epochs=i)
            if val_acc >= 0.85:
                manager.save()
            if val_acc >= 0.9:
                break

    try:
        train_acc, _ = evaluate_model(classifier, plain_train_dataset, 'train', None)
    except Exception as e:
        print(e)
        pass

    augmented_train_acc, _ = evaluate_model(classifier, train_dataset, 'augmented_train', None)  # augmented training set
    val_acc, val_top_k = evaluate_model(classifier, val_dataset, 'validation', None)
    test_acc, test_top_k = evaluate_model(classifier, test_dataset, 'test', None)
    with open('cifar-10_log.txt', 'a') as fo:
        fo.write('\nTraining accuracy: ' + str(train_acc.numpy()))
        fo.write('\nAugmented_raining accuracy: ' + str(augmented_train_acc.numpy()))
        fo.write('\nValidation accuracy: ' + str(val_acc.numpy()))
        fo.write('\nValidation top_k accuracy: ' + str(val_top_k.numpy()))
        fo.write('\nTest accuracy: ' + str(test_acc.numpy()))
        fo.write('\nTest top_k accuracy: ' + str(test_top_k.numpy()))
