import gym
import gym_ple

from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model, Sequential
from keras.optimizers import Adam

from dqn import DeepQNetwork


def make_model():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), input_shape=(64, 64, 3), strides=(4, 4), activation='relu'))
    model.add(Conv2D(32, (4, 4), input_shape=(64, 64, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3))
    model.compile(Adam(lr=0.0001, decay=1e-9), "mse")
    model.summary()
    return model


def main():
    env = gym.make('Catcher-v0')

    model = make_model()
    model.load_weights('data/weights520000.dat')

    net = DeepQNetwork(env, model, 10000)

    # net.train()
    net.play()
    net.play()
    net.play()


if __name__ == '__main__':
    main()
