import gym
import gym_ple

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from dqn import DeepQNetwork


def make_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5))
    model.compile(SGD(lr=0.001, clipnorm=1.), "mse")
    model.summary()
    return model


def main():
    env = gym.make('Snake-v0')
    model = make_model()

    net = DeepQNetwork(env, model, 10000)
    net.train()


if __name__ == '__main__':
    main()
