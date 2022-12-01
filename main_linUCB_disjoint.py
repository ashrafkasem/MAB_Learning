from MAP_core.utils.ctr_simulator import ctr_simulator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    alpha_input = 1.5
    data_path = "./data/news_dataset.txt"
    for alpha_input in [1.5,1.0,0.5]:

        aligned_time_steps, cum_rewards, aligned_ctr, policy = ctr_simulator(K_arms=10, d=100, alpha=alpha_input,
                                                                             data_path=data_path)
        # %%
        plt.plot(aligned_ctr)
        plt.title("alpha = " + str(alpha_input))
        plt.show()
