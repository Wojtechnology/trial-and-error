import datetime
import random
from collections import deque

from keras import backend as K
import numpy as np


MAX_CONSTANT = 1000


# TODO: Would be nice to gave a general abstraction for env, so that we are not just tied to the
# gym interface.


class Memory(object):

    def __init__(self, maxlen):
        self.experiences = deque(maxlen=maxlen)

    def add(self, experience):
        self.experiences.append(experience)

    def sample(self, size):
        return random.sample(self.experiences, size)

    def maxlen(self):
        return self.experiences.maxlen


class DeepQNetwork(object):

    def __init__(self, env, model, memory_len=MAX_CONSTANT):
        self.env = env
        self.model = model
        self.memory = Memory(memory_len)

    def train(
            self,
            gamma=0.9,
            epsilon=(1., 0.1),
            epsilon_decay_steps=100000,
            episodes=10000,
            batch_size=32):
        """
        Trains the model using the given hyperparameters.

        No state is reset before starting training.
        """
        model = self.model
        env = self.env
        memory = self.memory

        # Right now assume we want to fill up memory first.
        # TODO: maybe make into parameter after
        explore_steps = memory.maxlen()
        # TODO: Right now assumes linear decay
        e = epsilon[0]
        e_decay = (epsilon[1] - epsilon[0]) / epsilon_decay_steps
        steps = 0
        loss = 0.0
        q_max = 0.0

        for epi in range(1, episodes+1):
            done = False
            s = env.reset()

            start_steps = steps

            # TODO: Max steps per episode is probably desired
            while not done:
                if steps < explore_steps or random.random() < e:
                    a = env.action_space.sample()
                else:
                    q = model.predict(np.array([s]))
                    a = int(np.argmax(q[0]))

                next_s, r, done, _ = env.step(a)

                memory.add((s, a, next_s, r, done))
                s = next_s

                if steps >= explore_steps:
                    # Train model on random sample from
                    l, q_m = self._train_on_batch(gamma, batch_size)
                    loss += l
                    q_max += q_m

                    if steps % 100 == 0:
                        print('Steps: {}, Avg Loss: {}, Episode: {}, e: {}, q_max: {}'.format(
                            steps, loss/100, epi, e, q_max/100))
                        loss = 0.0
                        q_max = 0.0

                    if steps % 1000 == 0:
                        model.save_weights('data/weights{}.dat'.format(steps))

                steps += 1

                if steps == explore_steps:
                    q_measure_sample = memory.sample(1000)
                    q_measure_sample = np.array([s for s, _, _, _, _ in q_measure_sample])

                if steps >= explore_steps and steps % 1000 == 0:
                    q = model.predict(q_measure_sample)
                    print('Q measure: {}'.format(np.max(q, axis=1).mean()))

                if steps >= explore_steps and e > epsilon[1]:
                    e += e_decay

        model.save_weights('data/weights.dat')

    def _train_on_batch(self, gamma, batch_size):
        model = self.model
        memory = self.memory

        n_actions = model.layers[-1].output_shape[1]

        # TODO: slow af
        samples = self.memory.sample(batch_size)
        s, a, next_s, r, done = zip(*samples)
        s = np.array(s)
        a = np.array(a)
        next_s = np.array(next_s)
        r = np.array(r).repeat(n_actions).reshape((batch_size, n_actions))
        done = np.array(done).repeat(n_actions).reshape((batch_size, n_actions))

        all_s = np.concatenate([s, next_s])
        q = model.predict(all_s)

        delta = np.zeros((batch_size, n_actions))
        delta[np.arange(batch_size), a] = 1
        qsa = np.max(q[batch_size:], axis=1).repeat(n_actions).reshape((batch_size, n_actions))

        targets = (1 - delta) * q[:batch_size] + delta * (r + gamma * (1 - done) * qsa)
        loss = float(model.train_on_batch(s, targets))
        if np.isnan(loss):
            raise Exception('loss is nan')

        return loss, np.max(q[:batch_size], axis=1).mean()

    def play(self, visualize=True):
        model = self.model
        env = self.env

        done = False
        s = env.reset()
        while not done:
            q = model.predict(np.array([s]))
            a = int(np.argmax(q[0]))

            env.render()
            s, r, done, _ = env.step(a)
            print(r)
