from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.datasets import mnist
from keras.utils import np_utils


def get_data(name="power"):

    if name == "power":
        return get_power_data()
    elif name == "communities":
        return get_communities_data()
    elif name == "MNIST":
        return get_MNIST_data()
    else:
        return None

def partition(list_in, N):
    import random
    X_train = list_in[0]
    y_train = list_in[1]
    Xy_train = np.zeros((X_train.shape[0], X_train.shape[1] + 1), dtype=np.float32)
    for i in range(X_train.shape[0]):
        Xy_train[i] = np.append(X_train[i], y_train[i])

    #random.shuffle(Xy_train)
    Xy_partitioned = [Xy_train[i::N] for i in range(N)]
    X_train_partitioned= [np.array([Xy_partitioned[i][j][0:-1] for j in range(Xy_partitioned[i].shape[0])]) for i in range(N)]
    y_train_partitioned = [np.array([Xy_partitioned[i][j][-1] for j in range(Xy_partitioned[i].shape[0])]) for i in range(N)]

    y_train_partitioned = np.int64(y_train_partitioned)

    #for n in range(N):
    #    X_train_partitioned
    return X_train_partitioned, y_train_partitioned


def non_iid_generator(X_train_partitioned, y_train_partitioned, N, included_digits):
    import itertools
    X_train_partitioned_nonIID = list(X_train_partitioned)
    y_train_partitioned_nonIID = list(y_train_partitioned)

    for j in range(N):
        idx = np.array([i for i in range(y_train_partitioned_nonIID[j].shape[0]) if y_train_partitioned_nonIID[j][i]
                        in included_digits[j]])

        X_train_partitioned_nonIID[j] = np.array(list(itertools.compress(X_train_partitioned_nonIID[j],
                                                                  [i in idx for i in
                                                                   range(len(y_train_partitioned_nonIID[j]))])))

        y_train_partitioned_nonIID[j] = np.array(list(itertools.compress(y_train_partitioned_nonIID[j],
                                                                  [i in idx for i in
                                                                   range(len(y_train_partitioned_nonIID[j]))])))

    X_train_partitioned_nonIID = np.array(X_train_partitioned_nonIID)
    y_train_partitioned_nonIID = np.array(y_train_partitioned_nonIID)

    return X_train_partitioned_nonIID, y_train_partitioned_nonIID


def get_power_data():
    """
    Read the Individual household electric power consumption dataset
    """

    data = pd.read_csv('data/household_power_consumption.txt',
                       sep=';', low_memory=False)

    # Drop some non-predictive variables
    data = data.drop(columns=['Date', 'Time'], axis=1)

    print(data.head())

    # Replace missing values
    data = data.replace('?', np.nan)

    # Drop NA
    data = data.dropna(axis=0)

    # Normalize
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    # Goal variable assumed to be the first
    X = data.values[:, 1:].astype('float32')
    y = data.values[:, 0].astype('float32')

    # Create categorical y for binary classification with balanced classes
    y = np.sign(y+0.46)
    y[np.where(y == 0)] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    no_class = 2                 #binary classification

    return X_train, X_test, y_train, y_test, no_class


def get_communities_data():
    """
    Read the Communities and Crime dataset
    """

    data = None

    with open('data/communities.names', 'r') as f:
        columns = [x.split(' ')[1] for x in f.readlines() if "@attribute" in x]
        data = pd.read_csv('data/communities.data', names=columns)

        # Drop some non-predictive variables
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)

        # Replace missing values
        data = data.replace('?', np.nan)

        # Drop NA
        data = data.dropna(axis=1)

        X = data.values[:, :-1]

        # Goal variable assumed to be the last
        y = data.values[:, -1]

        # Create categorical y for binary classification with balanced classes
        y = np.sign(y-0.16)
        y[np.where(y == 0)] = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        no_class = 2                 #binary classification

        return X_train, X_test, y_train, y_test, no_class


def get_MNIST_data():
    """
    Read the MNIST dataset
    """

    '''
    import pandas as pd
    data = pd.read_csv('./data/mnist_train.csv', sep=',', low_memory=False)
    data = pd.DataFrame(data)
    X = data.values[:, 1:].astype('float32')
    y = data.values[:, 0].astype('int32')
    y
    '''

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np.int64(y_train)
    y_test = np.int64(y_test)

    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')


    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255
    no_class = 10                 #all 10 digits from 0 to 9

    return X_train, X_test, y_train, y_test, no_class