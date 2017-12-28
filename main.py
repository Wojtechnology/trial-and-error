import gym
import gym_ple

from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model, Sequential
from keras.optimizers import SGD

from dqn import DeepQNetwork


def make_model():
    model = Sequential()
    model.add(MaxPooling2D(input_shape=(64, 64, 3), pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3))
    model.compile(SGD(lr=0.001, clipnorm=1.), "mse")
    model.summary()
    return model


def main():
    env = gym.make('Catcher-v0')

    model = make_model()
    # model.load_weights('data/weights.dat')

    net = DeepQNetwork(env, model, 10000)

    net.train()
    # net.play()


if __name__ == '__main__':
    main()
