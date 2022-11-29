import random
import math

import numpy as np

from MAP_core.policies.policyABC import policyABC


class UCB1(policyABC):
    def __init__(self, *args, **kwargs):
        super(UCB1, self).__init__(*args, **kwargs)

    # UCB1 arm selection
    def select_arm(self):
        # UCB arm selection based on max of UCB reward of each arm
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ucb_values.index(max(ucb_values))
