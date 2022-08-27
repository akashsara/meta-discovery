class Epsilon:
    def __init__(self):
        pass

    def calculate_epsilon(self):
        raise NotImplementedError()

class LinearDecayEpsilon(Epsilon):
    """
    Epsilon is our exploration factor AKA the probability of picking Pokemon
    based on 1 - pickrate instead of winrate.
    We use two variables here - epsilon_max, epsilon_min
    epsilon_max is our starting epsilon value, epsilon_min is the final value.
    Epsilon Decay is the number of calls to generate_teams() over which epsilon
    moves from epsilon_max to epsilon_min.

    Linear Decay:
        epsilon = (-(max_eps - min_eps) * iterations) / max_steps + max_eps
        epsilon = max(min_eps, epsilon)
    """

    def __init__(self, max_epsilon: float, min_epsilon: float, epsilon_decay: float):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def calculate_epsilon(self, iterations: int) -> float:
        epsilon = (
            self.max_epsilon
            - ((self.max_epsilon - self.min_epsilon) * iterations) / self.epsilon_decay
        )
        return max(self.min_epsilon, epsilon)
