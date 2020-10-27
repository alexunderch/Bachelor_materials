import os
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import backend as K
from sklearn.model_selection import train_test_split


data_path = '../fashion-mnist/data/fashion/' #your way to work_dir
def open_fashion_mnist(data_path = data_path, test_size = 0.2):
    
    #retrieving dims
    fd = open(os.path.join(data_path, f'train-images-idx3-ubyte'), 'rb')
    load_dims_train = np.fromfile(file = fd, dtype = np.uint32).byteswap(inplace = True)
    tlength, trows, tcolumns = load_dims_train[1], load_dims_train[2].astype(np.uint8), load_dims_train[3].astype(np.uint8)
    fd = open(os.path.join(data_path, f't10k-images-idx3-ubyte'), 'rb')
    load_dims_test = np.fromfile(file = fd, dtype = np.uint32).byteswap(inplace = True)
    slength, srows, scolumns = load_dims_test[1], load_dims_test[2].astype(np.uint8), load_dims_test[3].astype(np.uint8)

    fd = open(os.path.join(data_path, f'train-images-idx3-ubyte'), 'rb')
    loaded = np.fromfile(file = fd, dtype = np.uint8).byteswap(inplace = True)
    X_train = loaded[16:].reshape((tlength, trows, tcolumns)).astype(float)
    
    fd = open(os.path.join(data_path, f't10k-images-idx3-ubyte'), 'rb')
    loaded = np.fromfile(file = fd, dtype = np.uint8).byteswap(inplace = True)
    X_test = loaded[16:].reshape((slength, srows, scolumns)).astype(float)
       
    fd = open(os.path.join(data_path, f'train-labels-idx1-ubyte'), 'rb')
    loaded = np.fromfile(file = fd, dtype = np.uint8).byteswap(inplace = True)
    Y_train = np.asarray(loaded[8:].reshape((tlength)))
    
    fd = open(os.path.join(data_path, f't10k-labels-idx1-ubyte'), 'rb')
    loaded = np.fromfile(file = fd, dtype = np.uint8).byteswap(inplace = True)
    Y_test = np.asarray(loaded[8:].reshape((slength)))
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = test_size, random_state = 0)
    
    return X_train, X_test, X_valid,  Y_train, Y_test, Y_valid

label_dict = {
 0: "T-shirt'/'top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

def scale_data(X, img_w = 28, img_h = 28):
    if K.image_data_format() == 'channels first':
        X = X.reshape(X.shape[0], 1, img_w, img_h)
        input_shape = (1, img_w, img_h)
    else:
        X = X.reshape(X.shape[0], img_w, img_h, 1)
    input_shape = (img_w, img_h, 1)

    X = X.astype('float32')
    X /= 255
    return X, input_shape

def recall_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score (y_true, y_pred):
    precision = precision_(y_true, y_pred)
    recall = recall_(y_true, y_pred)
    return 2 * ((precision * recall)/(precision + recall + K.epsilon()))

def _model(input_shape, num_classes = 10):
    model = Sequential([
    Conv2D(32, kernel_size = (3, 3), activation = 'elu', padding = 'same',
           input_shape = input_shape),
    Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    BatchNormalization(),
    Dropout(0.5),

    Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.5),

    Conv2D(128, kernel_size = (3, 3), activation = 'elu', padding = 'same'),
    Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.33),

    Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    Conv2D(256, kernel_size = (3, 3), activation = 'elu', padding = 'same'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.5),

    Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(pool_size = (2, 2)),
    Dropout(0.5),

    Flatten(),

    Dense(512, activation = 'relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation = 'relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation = 'softmax')
    ])
    
    model.compile(loss = categorical_crossentropy,
                  optimizer = keras.optimizers.adam(lr = 0.0015),
                  metrics = [precision_, recall_, f1_score])
    model.summary()
    return model
    