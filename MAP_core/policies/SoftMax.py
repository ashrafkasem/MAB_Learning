import random
import math

import numpy as np

from MAP_core.policies.policyABC import policyABC



# Arm selection based on Softmax probability
def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0

    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob

        if cum_prob > z:
            return i
    return len(probs) - 1


class SoftMax(policyABC):
    def __init__(self, *args, tau, **kwargs):
        self.tau = tau
        super(SoftMax, self).__init__(*args, **kwargs)

    # softmax arm selection
    def select_arm(self):
        # Calculate Softmax probabilities based on each round
        z = sum([math.exp(v/self.tau) for v in self.values])
        probs = [math.exp(v/self.tau) / z for v in self.values]
        return categorical_draw(probs)
