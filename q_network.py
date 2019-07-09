from keras import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, Reshape
from keras.optimizers import Adam


def atari_model(action_size, learning_rate, state_size):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=state_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(action_size, activation="linear"))

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model


def custom_model(action_size, learning_rate, state_size):
    model = Sequential()
    model.add(Reshape((64,), input_shape=state_size))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(action_size, activation="linear"))

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model


''''
def custom_cnn_model(action_size, learning_rate, state_size):
    model = Sequential()
    model.add(Convolution2D(8, 1, 2, subsample=(1, 2), border_mode='same', input_shape=state_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 1, 2, subsample=(1, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(action_size, activation="linear"))

    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model
'''