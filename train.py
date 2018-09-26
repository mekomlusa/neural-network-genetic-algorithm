"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import SGD
from keras.regularizers import l1_l2
import math
from keras import backend as K

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10(bs):
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = bs

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]
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
    # input image dimensions
    img_rows, img_cols = 28, 28

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
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
    if conv_layer_count > 0:
        for _ in range(conv_layer_count):
            # Need input shape for first layer.
            if len(model.layers) == 0:
                model.add(Conv2D(filters_per_conv, filter_size, activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))
                model.add(MaxPooling2D(pool_size=(2, 2)))  # hard-coded maxpooling
            elif model.layers[-1].output_shape[1] > filter_size[1] and model.layers[-1].output_shape[1] > 2:
                # valid, can subtract
                model.add(Conv2D(filters_per_conv, filter_size, activation='relu', kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))
                model.add(MaxPooling2D(pool_size=(2, 2)))  # hard-coded maxpooling
                        
        model.add(Flatten())
    
    # Then get hidden layers.
    if hidden_layer_count > 0:
        for _ in range(hidden_layer_count):
            if len(model.layers) == 0:
                # Need to add a flatten layer here
                model.add(Flatten())
                model.add(Dense(units_per_hidden, activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))
            else:
                model.add(Dense(units_per_hidden, activation='relu', kernel_regularizer=l1_l2(l1=l1_penalty,l2=l2_penalty)))

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))
    #print(model.summary())

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
    
    tbCallBack = TensorBoard(log_dir='./Graph/CIFAR10_RS', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,  # per paper
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper, tbCallBack])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

def train_and_score_TB(network, dataset, iteration, current_network_count, dataset_TB_folder_name):
    """Train the model, return test loss.
    Special cases for tensorboard for multiple runs.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
        iteration (int): Count of the current iteration.
        current_network_count (int): Count of the current network.
        dataset_TB_folder_name (str): Name of the parent folder that holds the multiple run tensorboard result.

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10(network['batch_size'])
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist(network['batch_size'])

    model = compile_model(network, nb_classes, input_shape)
    
    tbCallBack = TensorBoard(log_dir='./Graph/'+dataset_TB_folder_name+'/Run'+str(iteration)+'/Model'+str(current_network_count)+'/', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=50,  # per paper
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper, tbCallBack])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.