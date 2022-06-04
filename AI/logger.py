import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self, save_dir):
        save_dir.mkdir(parents=True)
        self.save_dir = save_dir
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Moves':>8}{'Epsilon':>10}{'MeanRewardThisEpisode':>15}"
                f"{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        self.winLossRatio_plot = save_dir / "winLossRatio_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.winLossRatio = []

        self.record_time = time.time()
        self.resetVariablesBeforeNewEpisode()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_loss += loss
        self.curr_ep_q += q
        self.steps += 1 # count number of actions this episode

    def log_episode(self, episode, epsilon, wins, losses):
        calculatedWinLossRatio = 0.5 if wins+losses == 0 else wins/(wins+losses)
        self.winLossRatio.append(calculatedWinLossRatio)
        meanReward = self.curr_ep_reward / self.steps
        meanLoss = self.curr_ep_loss / self.steps
        meanQ = self.curr_ep_q / self.steps

        self.ep_rewards.append(meanReward)
        self.ep_avg_losses.append(meanLoss)
        self.ep_avg_qs.append(meanQ)

        # update time
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # print
        print(
            f"Episode {episode} - "
            f"Steps this episode {self.steps} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {meanReward} - "
            f"Mean Loss {meanLoss} - "
            f"Mean Q Value {meanQ} - "
            f"Time Delta {time_since_last_record} - "
            f"Win Loss Ratio {calculatedWinLossRatio}"
        )

        # write logs
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{self.steps:8d}{epsilon:10.3f}"
                f"{meanReward:15.3f}{meanLoss:15.3f}{meanQ:15.3f}"
                f"{time_since_last_record:15.3f}\n"
            )

        # plots
        for metric in ["ep_rewards", "ep_avg_losses", "ep_avg_qs", "winLossRatio"]:
            plt.plot(getattr(self, f"{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

        self.resetVariablesBeforeNewEpisode()

    def resetVariablesBeforeNewEpisode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.steps = 0