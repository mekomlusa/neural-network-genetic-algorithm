"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l1_l2

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10(bs):
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = bs
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist(bs):
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = bs
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.    
    filter_size = network['filter_size']
    l1_penalty = network['l1_penalty']
    l2_penalty = network['l2_penalty']
    learning_rate = network['learning_rate']
    conv_layer_count = network['conv_layer_count']
    filters_per_conv = network['filters_per_conv']
    hidden_layer_count = network['hidden_layer_count']
    units_per_hidden = network['units_per_hidden']

    model = Sequential()

    # Add each layer.
    # Arrange conv layers first.
    for i in range(conv_layer_count):

        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(filters_per_conv, filter_size, activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))
        else:
            model.add(Conv2D(filters_per_conv, filter_size, activation='relu', kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))

        model.add(MaxPooling2D(pool_size=(2, 2)))  # hard-coded maxpooling
    
    # Then get hidden layers.
    for i in range(hidden_layer_count):
        model.add(Dense(units_per_hidden, activation='relu', kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate, momentum=0.9),
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10(network['batch_size'])
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist(network['batch_size'])

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,  # per paper
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
