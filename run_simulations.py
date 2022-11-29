

def test_algorithm(algo, arms, num_sims, horizon):
    """
    :param algo: chosen algorithm to perform the simulations
    :param arms: environment from which we draw the arm distribution
    :param num_sims: Represents the number of independent simulations, each of length equal to 'horizon'.
    :param horizon: Represents the number of time steps/trials per round of simulation
    :return: [sim_nums, times, chosen_arms, rewards, cumulative_rewards]
    """
    # Initialise variables for duration of accumulated simulation (num_sims * horizon_per_simulation)
    chosen_arms = [0 for i in range(num_sims * horizon)]
    rewards = [0.0 for i in range(num_sims * horizon)]
    cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
    sim_nums = [0 for i in range(num_sims * horizon)]
    times = [0 for i in range(num_sims * horizon)]
    for sim in range(num_sims):

        sim += 1

        algo.initialize(len(arms))

        for t in range(horizon):
            t += 1
            index = (sim - 1) * horizon + (t - 1)
            sim_nums[index] = sim
            times[index] = t
            # Selection of best arm and engaging it
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            # Engage chosen Bernoulli Arm and obtain reward info
            reward = arms[chosen_arm].draw()
            rewards[index] = reward
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]
