import numpy as np
import scipy.signal as signal
import scipy.stats as sts
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Lambda
from keras.models import Sequential

class generator:
    def __init__(self, parameters):
        self.__len__ = 1024
        self.sgnl_type, self.tmp = parameters
        self.signal_ = np.zeros(self.__len__); self.t = np.linspace(0, self.__len__, self.__len__) 
        """
        signal types: 
        determined:
            -- periodic signal with frequency f (f = 1 / T where T is period) and amplitude A; ph is #pi-fold.
            -- aperiodic signal with main frequency f and time tau;
            -- non-periodic signals like exponent
        random:
            -- stationary -- gaussian or bernoulli 
            -- non-stationary -- full random
        """
        if self.sgnl_type == "periodic":
            self.A, self.f, self.ph = self.tmp
            self.signal_ = self.A * np.sin(2 * np.pi * self.f * self.t / self.__len__  + np.pi * self.ph)
        
        elif self.sgnl_type == "aperiodic":
            self.A, self.f, self.ph, self.tau = self.tmp
            self.signal_ = self.A * np.sin(2 * np.pi * self.f * self.t / self.__len__ + np.pi * self.ph) * np.exp(- self.t/self.tau)

        elif self.sgnl_type == "non-periodic":
            self.A, self.tau = self.tmp
            self.signal_ = self.A * np.exp(- self.t/self.tau)

        elif self.sgnl_type == "stationary":
            self.A = self.tmp
            self.signal_ = self.A * sts.norm.rvs(size = self.__len__)
        elif self.sgnl_type == "non-stationary":
            self.A = self.tmp
            self.w = 0.23456
            self.signal_ = self.A * (self.w * sts.norm.rvs(size = self.__len__) + (1 - self.w) * sts.bernoulli.rvs(p = self.w, size = self.__len__))
        elif self.sgnl_type == "mixed":
            self.A, self.f, self.ph, self.tau = self.tmp
            self.w = 0.01
            self.signal_ = self.A * self.w * sts.norm.rvs(size = self.__len__) + \
             (1 - self.w) * self.A * np.sin(2 * np.pi * self.f * self.t / self.__len__  + np.pi * self.ph) * np.exp(- self.t/self.tau)

    def _make_signal(self):
        return self.signal_, signal.periodogram(self.signal_)[1]

def get_dataset(n_samples = 2000):
    signal_ = []
    for _ in range(n_samples):
        A = random.uniform(0.0, 20.0)
        f = random.uniform(0.01, 100.0)
        ph = random.uniform(-np.pi /2 , np.pi /2)
        tau = random.uniform(0.1, 5.0)
        signal_.append(generator(("periodic", (A, f, ph)))._make_signal())
        signal_.append(generator(("aperiodic", (A, f, ph, tau)))._make_signal())
        signal_.append(generator(("non-periodic", (A, tau)))._make_signal())
        signal_.append(generator(("stationary", A))._make_signal())
        signal_.append(generator(("non-stationary", A))._make_signal())
        signal_.append(generator(("mixed", (A, f, ph, tau)))._make_signal())

    X, y =  np.array([a[0] for a in signal_]), np.array([a[1] for a in signal_])
    a, b = X.shape[0], X.shape[1]
    return X.reshape((a, b, 1)), y

def evaluate_model(X_train, y_train, X_valid, y_valid):
    verbose, epochs, batch_size = 1, 2000, 200
  
    model = Sequential([Input(shape = (1024, 1)), 
                        Flatten(),
                        Lambda(lambda v: tf.cast(tf.signal.rfft(tf.cast(v, dtype = tf.float32)), tf.float32)),
                        Dense(513, use_bias = False)])
  
    model.compile(loss = 'mean_absolute_error', optimizer = 'adadelta')
    # fit network
    print(model.summary())
    history = model.fit(X_train, y_train, validation_data = (X_valid, y_valid), 
                        epochs = epochs, batch_size = batch_size, verbose = verbose)
   
    return model, history


def __main__(): 
    np.random.seed(0)
    X_train, y_train = get_dataset(20000)
    X_valid, y_valid = get_dataset(5000)
    model, history = evaluate_model(X_train, y_train, X_valid, y_valid)
    plt.plot(history.history['loss']); plt.show()
    plt.plot(history.history['val_loss']); plt.show()
    X_test, y_test = get_dataset(1)
    y_pred = model.predict(X_test)
    print(model.evaluate(X_test, y_test))
    for ind in range(len(y_test)):
      plt.plot(X_test[ind]); plt.show()
      plt.plot(y_test[ind]); plt.show()
      plt.plot(y_pred[ind]); plt.show()


if __name__ == "__main__":
    __main__()