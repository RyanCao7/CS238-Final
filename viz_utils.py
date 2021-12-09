import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import json
sns.set_theme()

import constants


def plot_losses_vps(train_stats_dict, viz_path):
    """
    Plots and saves.
    """

    def mean(it):
        if len(it) == 0: return 0
        return sum(it) / len(it)

    losses = train_stats_dict['smoothed_losses'][:400]
    loss_num_steps = list(x * 500 for x in range(len(losses)))

    vps = train_stats_dict['VPs']
    avg_vps = list()

    # --- Window size is 10 ---
    for idx in range(len(vps)):
        small_idx = max(0, idx - 10)
        large_idx = min(len(vps) - 1, idx + 10)
        avg_vp = mean(vps[small_idx : large_idx])
        avg_vps.append(avg_vp)
    avg_vps_num_episodes = list(range(len(avg_vps)))

    loss_path = os.path.join(viz_path, 'train_losses.png')
    print(f'Saving losses to {loss_path}...')
    loss_plot = sns.lineplot(x=loss_num_steps, y=losses)
    plt.title('Average Train Loss Over Timesteps')
    plt.ylabel('Average Huber DQN Loss')
    plt.xlabel('Number of timesteps')
    fig = loss_plot.get_figure()
    fig.savefig(loss_path)
    plt.clf()

    vps_path = os.path.join(viz_path, 'train_vps.png')
    print(f'Saving VPs to {vps_path}...')
    iou_plot = sns.lineplot(x=avg_vps_num_episodes, y=avg_vps)
    plt.title('Avg Train Victory Points / Episode')
    plt.ylabel('Number of VPs')
    plt.xlabel('Number of Episodes')
    fig = iou_plot.get_figure()
    fig.savefig(vps_path)
    plt.clf()


if __name__ == '__main__':
    model_type = 'Catan_Feedforward_DQN'
    model_name = 'ryan_test_2'
    viz_path = os.path.join(constants.get_viz_save_dir(model_type, model_name))
    save_path = os.path.join(constants.get_model_save_dir(model_type, model_name), 'train_stats.json')
    with open(save_path, 'r') as f:
        train_stats_dict = json.load(f)
    plot_losses_vps(train_stats_dict, viz_path)