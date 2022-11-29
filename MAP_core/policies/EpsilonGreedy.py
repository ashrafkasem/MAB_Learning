import random

import numpy as np

from MAP_core.policies.policyABC import policyABC

class EpsilonGreedy(policyABC):
    def __init__(self, *args, epsilon, **kwargs):
        self.epsilon = epsilon
        super(EpsilonGreedy, self).__init__(*args, **kwargs)

    # Epsilon greedy arm selection
    def select_arm(self):
        # If prob is not in epsilon, do exploitation of best arm so far
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        # If prob falls in epsilon range, do exploration
        else:
            return random.randrange(len(self.values))
