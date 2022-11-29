import random
import pandas as pd
import altair as alt
import numpy as np
import altair_viewer

from MAP_core.envs.BernoulliArm import BernoulliArm
from MAP_core.policies.SoftMax import SoftMax
from run_simulations import test_algorithm

random.seed(1)


if __name__ == '__main__':
    # out of 5 arms, 1 arm is clearly the best
    similar_arms = 0.1
    best_arms = 0.9
    n_similar_arms = 4
    n_best_arms = 1

    means = n_similar_arms * [similar_arms] + n_best_arms * [best_arms]
    #[0.1, 0.1, 0.1, 0.1, 0.9]
    #
    n_arms = len(means)
    # Shuffling arms
    random.shuffle(means)

    # Create list of Bernoulli Arms with Reward Information
    arms = list(map(lambda mu: BernoulliArm(mu), means))
    print("Best arm is " + str(np.argmax(means)))
    f = open(f"standard_softmax_results_{similar_arms}_{best_arms}.tsv", "w+")

    # Create simulations for each exploration epsilon value
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = SoftMax(tau=tau, counts=[], values=[])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, 5000, 250)
        # store data
        for i in range(len(results[0])):
            f.write(str(tau) + "\t")
            f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()
    print("done!")
    df = pd.read_csv(f"standard_softmax_results_{similar_arms}_{best_arms}.tsv", sep = "\t", header = None, names = ["tau", "simulation_num", "step", "chosen_arm", "reward", "cum_reward"])

    # Create a flag for each step to see if best arm was chosen
    df["chose_correct"] = np.select(
        [
            df["chosen_arm"] == 2,
            df["chosen_arm"] != 2
        ],
        [
            1,
            0
        ]
    )
    # Perform average/mean for each step for all simulations and epsilon
    df_chose_correctly = df.loc[:, ["tau", "step", "chose_correct"]].groupby(["tau", "step"]).agg("mean")

    # Remove multi index grouping
    df_chose_correctly = df_chose_correctly.reset_index()
    print(df_chose_correctly.head())
    chart = alt.Chart(df_chose_correctly).mark_line().encode(
        alt.X("step:Q", title = "Time Step"),
        alt.Y("chose_correct:Q", title = "Mean Rate of Choosing Best Arm", scale = alt.Scale(domain = [0, 1])),
        color = alt.Color("tau:N")
    ).properties(
        title = f"Softmax: Mean Rate of Choosing Best Arm from 5000 Simulations. {n_best_arms+n_similar_arms} Arms = [{n_similar_arms} x {similar_arms}, {n_best_arms} x {best_arms}]",
    )

    altair_viewer.show(chart)

    df_cumreward = df.loc[:, ["tau", "step", "cum_reward"]].groupby(["tau", "step"]).agg("mean").reset_index()

    chart2 = alt.Chart(df_cumreward).mark_line().encode(
        alt.X("step:Q", title = "Time Step"),
        alt.Y("cum_reward:Q", title = "Mean Cumulative Reward"),
        color = alt.Color("tau:N")
    ).properties(
        title = f"Softmax: Mean Cumulative Reward from 5000 Simulations. {n_best_arms+n_similar_arms} Arms = [{n_similar_arms} x {similar_arms}, {n_best_arms} x {best_arms}]",
    )

    altair_viewer.show(chart2)

    df_cumreward["best_cumreward"] = df["step"] * max(means)
    df_cumreward["regret"] = df_cumreward["best_cumreward"] - df_cumreward["cum_reward"]

    chart3 = alt.Chart(df_cumreward).mark_line().encode(
        alt.X("step:Q", title="Time Step"),
        alt.Y("regret:Q", title="Mean Cumulative Regret"),
        alt.Color("tau:N")
    ).properties(
        title=f"Softmax: Mean Cumulative Regret from 5000 Simulations. {n_best_arms+n_similar_arms} Arms = [{n_similar_arms} x {similar_arms}, {n_best_arms} x {best_arms}]",
    )

    altair_viewer.show(chart3)