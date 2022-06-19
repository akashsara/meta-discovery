import numpy as np
import numpy.random as random


class EpsilonGreedyPolicy:
    """
    Base class for an Epsilon Greedy Policy.
    Does not implement calculate_epsilon.
    The Epsilon Greedy Policy:
        {random action, epsilon probability}
        {greedy action, 1 - epsilon probability}
    """

    def __init__(self):
        self.iterations = 0

    def calculate_epsilon(self):
        raise NotImplementedError()

    def select_action(self, q_values, action_mask=None):
        nb_actions = q_values.shape[0]
        if random.random() < self.calculate_epsilon():
            if action_mask is not None:
                # Move mask to cpu since we're using numpy for action selection
                # Torch doesn't have a native random.choice
                action_mask = action_mask.cpu()
                action = random.choice((action_mask == 0).nonzero().flatten())
            else:
                action = random.randint(0, nb_actions)
        else:
            if action_mask is not None:
                q_values += action_mask
            action = q_values.argmax()
        self.iterations += 1
        return action

    def greedy_action(self, q_values, action_mask=None):
        if action_mask is not None:
            q_values += action_mask
        return q_values.argmax()


class LinearDecayEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    """
    Linear Decay:
        epsilon = (-(max_eps - min_eps) * iterations) / max_steps + max_eps
        epsilon = max(min_eps, epsilon)
    """

    def __init__(self, max_epsilon, min_epsilon, max_steps):
        super(LinearDecayEpsilonGreedyPolicy, self).__init__()
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.max_steps = max_steps

    def calculate_epsilon(self):
        x = -(self.max_epsilon - self.min_epsilon) * self.iterations
        x = (x / self.max_steps) + self.max_epsilon
        return max(self.min_epsilon, x)


class ExponentialDecayEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    """
    Exponential Decay:
        epsilon = eps_end + (eps_start - eps_end) * exp(-iterations/decay)
        epsilon = max(epsilon, x)
    """

    def __init__(self, max_epsilon, min_epsilon, epsilon_decay):
        super(ExponentialDecayEpsilonGreedyPolicy, self).__init__()
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def calculate_epsilon(self):
        x = self.min_epsilon + (self.max_epsilon - self.min_epsilon)
        x = x * np.exp(-1.0 * self.iterations / self.epsilon_decay)
        return max(self.min_epsilon, x)
