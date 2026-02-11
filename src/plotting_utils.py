"""
Plotting utilities for execution results analysis.

Contains dedicated plotting functions for all plots created in the 
execution_results_analysis.ipynb notebook. Each function supports 
optional saving and showing of plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


FIGURES_DIR = '../figures/'


# --------------------------------------------------------------------------
# Baseline plots
# --------------------------------------------------------------------------

def plot_baseline_results(baseline_results, episode_id=-1, ma_window=0, figsize=None,
                          date_id=None, save=False, show=True):
    """
    Plot baseline results overview with rewards, terminal inventories, 
    actions, and inventories in a 2x2 subplot grid.

    Args:
        baseline_results (dict): Baseline results dictionary with keys 
            'rewards', 'terminal_inventories', 'actions', 'inventories'.
        episode_id (int): Episode index to display for actions/inventories. 
            Defaults to -1 (last episode).
        ma_window (int): Window size for simple moving average overlay. 
            Defaults to 0 (no SMA).
        figsize (tuple or None): Figure size (width, height). Defaults to None.
        date_id (str or None): Date identifier appended to saved file name.
            Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    names = ["rewards", "terminal_inventories", "actions", "inventories"]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=figsize)
    for ax, id in zip(axes.flat, range(4)):
        ax.set_title(names[id] + f' in baseline run' if id < 2 else names[id] + (
                        " in episode " + str(episode_id) if episode_id != -1 else " in last episode")
        )
        if episode_id is not None:
            ax.plot(baseline_results[names[id]] 
            if id < 2 else 
            baseline_results[names[id]][episode_id])
        else:
            ax.plot(baseline_results[names[id]] 
            if id < 2 else 
            [ax.plot(x) for x in baseline_results[names[id]]])

        if ma_window > 0:
            ax.plot(
                pd.Series(baseline_results[names[id]]).rolling(ma_window).mean()
                if id < 2 else
                pd.Series(baseline_results[names[id]][episode_id]).rolling(ma_window).mean(), 
                label="SMA {}".format(ma_window)
            )             
        ax.set_xlabel("Episodes" if id < 2 else "Steps")
    fig.tight_layout()

    file_name = f'baseline_results_overview_{date_id}.png' if date_id else 'baseline_results_overview.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_baseline_trajectories(baseline_results, episode_ids=[-1], rho=50,
                               figsize=(5.5, 4.125), date_id=None, save=False, show=True):
    """
    Plot baseline action and inventory trajectories for selected episodes.
    Creates two separate figures: one for actions, one for inventories.

    Args:
        baseline_results (dict): Baseline results dictionary with keys 
            'actions' and 'inventories'.
        episode_ids (list of int): List of episode indices to plot. 
            Maximum 4 episodes. Defaults to [-1].
        rho (int): Inventory threshold; horizontal lines at +/- rho/2 
            are drawn. Defaults to 50.
        figsize (tuple): Figure size (width, height). Defaults to (5.5, 4.125).
        date_id (str or None): Date identifier appended to saved file names.
            Defaults to None.
        save (bool): Whether to save the plots. Defaults to False.
        show (bool): Whether to show the plots. Defaults to True.

    Returns:
        None
    """
    assert len(episode_ids) <= 4, "Maximum of 4 episodes can be plotted."

    linestyles = ['-', '--', '-.', ':'][0:len(episode_ids)]
    plt.rcParams['text.usetex'] = False
    steps = len(baseline_results["inventories"][0])

    # actions
    plt.figure(figsize=figsize)
    for episode, style in zip(episode_ids, linestyles):
        plt.plot(baseline_results["actions"][episode], lw=2, ls=style)
    plt.xlabel("Step", fontsize=11)
    plt.xticks([0 + 20*i for i in range(10)], fontsize=11)
    plt.ylabel("Action", fontsize=11)
    plt.yticks([0, 1, 2], fontsize=11)
    plt.legend(["Episode one", "Episode two"], 
               loc="lower right", fontsize=11)
    plt.tight_layout()
    file_name_actions = f'baseline_actions_{date_id}.png' if date_id else 'baseline_actions.png'
    if save:
        plt.savefig(FIGURES_DIR + file_name_actions)
        print(f"Plot saved under '{FIGURES_DIR + file_name_actions}'")
    if show:
        plt.show()
    else:
        plt.close()

    # inventories
    plt.figure(figsize=figsize)
    for episode, style in zip(episode_ids, linestyles):
        plt.plot(baseline_results["inventories"][episode], lw=2, ls=style)
    plt.plot([rho//2 for _ in range(steps)], ls='dotted', color='black', lw=1.5)
    plt.plot([-rho//2 for _ in range(steps)], ls='dotted', color='black', lw=1.5)
    plt.xlabel("Step", fontsize=11)
    plt.xticks([0 + 20*i for i in range(steps//20+1)], fontsize=11)
    plt.ylabel("Inventory", fontsize=11)
    plt.legend(["Episode one", "Episode two", r"$\pm\,\rho\, /\, 2$"], 
               loc="best", fontsize=11)
    plt.tight_layout()
    file_name_inventories = f'baseline_inventories_{date_id}.png' if date_id else 'baseline_inventories.png'
    if save:
        plt.savefig(FIGURES_DIR + file_name_inventories)
        print(f"Plot saved under '{FIGURES_DIR + file_name_inventories}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_baseline_reward_distribution(baseline_rewards, figsize=(10, 3.5),
                                      date_id=None, save=False, show=True):
    """
    Plot histogram and boxplot of baseline rewards side by side.

    Args:
        baseline_rewards (np.ndarray): Array of baseline reward values.
        figsize (tuple): Figure size (width, height). Defaults to (11, 3.5).
        date_id (str or None): Date identifier appended to saved file name.
            Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    baseline_rewards = np.asarray(baseline_rewards)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].hist(baseline_rewards, bins=80, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0].set_title('Baseline rewards histogram', fontsize=11)
    axs[0].set_xlabel('Rewards')
    axs[0].set_ylabel('Frequency')
    bp = axs[1].boxplot(baseline_rewards, showmeans=True, meanline=True, patch_artist=True, 
                        boxprops=dict(facecolor="white"))
    axs[1].legend([bp['means'][0], bp['medians'][0], bp['boxes'][0], bp['whiskers'][0]], 
                  ['mean', 'median', 'IQR', r'$\pm$1.5 IQR'], loc='lower right')
    axs[1].set_ylabel('Rewards')
    axs[1].set_xticks([1], [])
    axs[1].set_title('Baseline rewards boxplot', fontsize=11)

    file_name = f'baseline_reward_distribution_{date_id}.png' if date_id else 'baseline_reward_distribution.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_baseline_lognormal_fit(baseline_rewards, shape, loc, scale,
                                  figsize=(10, 7), date_id=None, save=False, show=True):
    """
    Visual inspection of the log-normal fit to negative baseline rewards.
    Creates a 2x2 subplot grid with histograms and Q-Q plots comparing 
    the fitted log-normal distribution to the observed data.

    Args:
        baseline_rewards (np.ndarray): Array of baseline reward values (negative).
        shape (float): Shape parameter of the fitted log-normal distribution.
        loc (float): Location parameter of the fitted log-normal distribution.
        scale (float): Scale parameter of the fitted log-normal distribution.
        figsize (tuple): Figure size (width, height). Defaults to (12, 9).
        date_id (str or None): Date identifier appended to saved file name.
            Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    baseline_rewards = np.asarray(baseline_rewards)
    n = len(baseline_rewards)

    samples = stats.lognorm.rvs(shape, loc, scale, size=n)
    stdzd_samples = (np.log(samples - loc) - np.log(scale)) / shape
    stdized_rewards = (np.log(-baseline_rewards - loc) - np.log(scale)) / shape

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    # axs 0
    axs[0].hist(samples, bins=80, color='blue', edgecolor='black', alpha=1)
    axs[0].hist(-baseline_rewards, bins=80, color='orange', edgecolor='black', alpha=0.8)
    axs[0].set_title('Histogram rewards and samples')
    axs[0].legend(['log-normal samples', 'negative baseline rewards'])
    # axs 1
    stats.probplot(-baseline_rewards, dist=stats.lognorm, sparams=(shape, loc, scale), plot=axs[1])
    axs[1].set_title('Q-Q plot negative rewards vs log-normal fit')
    # axs 2
    axs[2].hist(stdzd_samples, bins=80, color='blue', edgecolor='black', alpha=1)
    axs[2].hist(stdized_rewards, bins=80, color='orange', edgecolor='black', alpha=0.8)
    axs[2].set_title('Histogram standardized rewards and samples')
    axs[2].legend(['log-normal samples', 'negative baseline rewards'])
    # axs 3
    stats.probplot(stdized_rewards, dist=stats.norm, plot=axs[3])
    axs[3].set_title('Q-Q plot stdzd rewards vs standard normal')

    file_name = f'baseline_lognormal_fit_{date_id}.png' if date_id else 'baseline_lognormal_fit.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


# --------------------------------------------------------------------------
# Training plots
# --------------------------------------------------------------------------

def plot_training_run_results(training_results, run_id, ma_window=50, figsize=None,
                              date_id=None, save=False, show=True):
    """
    Plot training results for a single run with rewards, losses, cash, 
    and terminal inventory in a 2x2 subplot grid.

    Args:
        training_results (dict): Training results dictionary keyed by run id.
        run_id (int): Run index to plot.
        ma_window (int): Window size for simple moving average overlay. 
            Defaults to 50.
        figsize (tuple or None): Figure size (width, height). Defaults to None.
        date_id (str or None): Date identifier appended to saved file name.
            Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    results_run = training_results[run_id]
    keys = list(results_run.keys())[0:4]  # rewards, losses, cash, terminal_inventory
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=figsize)
    for ax, y_label, count in zip(axes.flat, keys, range(4)):
        ax.plot(results_run[y_label])
        ma = ax.plot(pd.Series(results_run[y_label]).rolling(ma_window).mean(), 
                     label="SMA {}".format(ma_window))
        ax.set_ylabel(y_label)
        if count == 1:
            fig.legend(handles=ma, loc='outside upper right')
        if count > 1:
            ax.set_xlabel('Episodes')

    fig.suptitle('Training results for run {}'.format(run_id), size=15)
    fig.tight_layout()

    file_name = f'training_run_{run_id}_results_{date_id}.png' if date_id else f'training_run_{run_id}_results.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_training_results(training_results, title=None, ma_window=50, figsize=None,
                               date_id=None, save=False, show=True):
    """
    Plot mean training results (reward, loss, cash, terminal inventory) 
    averaged over all runs, with +/- one standard deviation bands, in a 
    2x2 subplot grid.

    When save=True, each of the four subplots is additionally saved as a 
    separate file (using the save logic from the former 
    save_mean_training_results_plots function).

    Args:
        training_results (dict): Training results dictionary keyed by run id.
        title (str or None): Suptitle for the combined 2x2 plot. Defaults to None.
        ma_window (int): Window size for simple moving average. Defaults to 50.
        figsize (tuple or None): Figure size for the combined 2x2 plot. 
            Defaults to None.
        date_id (str or None): Date identifier appended to saved file names. 
            Required when save=True.
        save (bool): Whether to save plots. Defaults to False.
        show (bool): Whether to show the combined plot. Defaults to True.

    Returns:
        None
    """
    # calculate mean and std over runs
    plot_keys = list(next(iter(training_results.values())).keys())[0:4]
    results_array = np.array([
        [run_results[key] for key in plot_keys] 
        for run_results in training_results.values()
    ])
    means = np.mean(results_array, axis=0)
    stds = np.std(results_array, axis=0)
    
    y_labels = ['reward', 'loss', 'cash', 'terminal inventory']

    # --- combined 2x2 plot ---
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained', figsize=figsize)
    for ax, y_label, id in zip(axes.flat, y_labels, range(4)):
        ax.plot(means[id], color="b", label="mean over runs") 
        ax.plot(pd.Series(means[id]).rolling(ma_window).mean())

        fill_lower = means[id] - stds[id] if y_label != 'loss' else [
            mean - std if mean - std >= 0. else 0 for mean, std in zip(means[id], stds[id])
        ]
        fill_upper = means[id] + stds[id]
        ax.fill_between(range(len(means[id])), fill_lower, fill_upper,
                         color='b', alpha=0.2, label='+/- one standard deviation')        
        ax.set_ylabel('mean ' + y_label)
        if id > 1:
            ax.set_xlabel('episodes')
        if id == 0:
            fig.legend(bbox_to_anchor=(0.18, 1.02, 1., .102), loc='lower left',
                      ncols=2, borderaxespad=-0.2)
    fig.suptitle(title, y=1.127)

    combined_file_name = f'training_mean_results_all_{date_id}.png' if date_id else 'training_mean_results_all.png'
    if save:
        fig.savefig(FIGURES_DIR + combined_file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + combined_file_name}'")
    if show:
        plt.show()
    else:
        plt.close()

    # --- individual plots for saving ---
    if save:
        assert date_id is not None, "Provide date_id to save individual mean training plots."
        save_figsize = (5.5, 4.125)
        for y_label, id in zip(y_labels, range(4)):
            plt.figure(figsize=save_figsize)        
            plt.plot(means[id], color="b", label='mean {}'.format(y_label))
            plt.plot(pd.Series(means[id]).rolling(ma_window).mean())
            fill_lower = means[id] - stds[id] if y_label != 'loss' else [
                mean - std if mean - std >= 0. else mean for mean, std in zip(means[id], stds[id])
            ]
            fill_upper = means[id] + stds[id]
            plt.fill_between(range(len(means[id])), fill_lower, fill_upper,
                             color='b', alpha=0.2, label='+/- one standard deviation')
            plt.ylabel(y_label, fontsize=11)
            plt.xlabel('episode', fontsize=11)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)        
            plt.legend(loc='best')
            plt.title(' ', fontsize=12)
            plt.tight_layout()
            individual_name = f'training_mean_{y_label.replace(" ", "_")}_{date_id}.png'
            plt.savefig(FIGURES_DIR + individual_name)
            plt.close()
            print(f"Plot saved under '{FIGURES_DIR + individual_name}'")



def plot_inventory_action_histories(training_results_dict, run_ids=0, episode_ids=-1, 
                                    n_actions=3, figsize=None, date_id=None,
                                    save=False, show=True):
    """
    Plot observation (inventory) and action trajectories for given run_ids 
    and episode_ids. Each row of subplots corresponds to one episode, 
    showing inventory and action histories for all selected runs.

    Args:
        training_results_dict (dict): Training results dictionary keyed by run id.
        run_ids (int, list of int, or 'all'): Run indices to plot. 
            Defaults to 0.
        episode_ids (int or list of int): Episode indices to plot. 
            Defaults to -1.
        n_actions (int): Total number of discrete actions. Defaults to 3.
        figsize (tuple or None): Figure size. Defaults to None (auto-scaled).
        date_id (str or None): Date identifier appended to saved file name.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    if type(run_ids) == str and run_ids == 'all': 
        run_ids = list(training_results_dict.keys())
    elif type(run_ids) == int:
        run_ids = [run_ids]
    if type(episode_ids) == int:
        episode_ids = [episode_ids]
    
    plot_rows = len(episode_ids)
    max_episodes = len(training_results_dict[0]["observations"])
    if figsize is None:
        figsize = (9, 2.5 * plot_rows)
    for i in range(plot_rows):
        episode_ids[i] = episode_ids[i] if episode_ids[i] >= 0 else max_episodes + episode_ids[i]

    fig, axs = plt.subplots(plot_rows, 2, sharex=True, figsize=figsize, squeeze=False)
    for episode_id, ax_row in zip(episode_ids, axs):
        for run_id in run_ids:
            observation_history = list(training_results_dict[run_id]["observations"][episode_id])
            action_history = training_results_dict[run_id]["actions"][episode_id]
            ax_row[0].plot([inventory for (_, inventory) in observation_history])
            ax_row[0].set_title(f'Episode {episode_id+1}', fontsize=10)
            ax_row[1].plot(action_history, label='Run {}'.format(run_id+1))
            ax_row[1].set_title(f'Episode {episode_id+1}', fontsize=10)

        if episode_id == episode_ids[0]:
            fig.legend(bbox_to_anchor=(0.22, 0.94, 1., .102), loc='center left',
                        ncols=5, borderaxespad=-0.2)

    actions = [*range(n_actions)]
    for ax_row in axs:
        ax_row[0].set_ylabel(r"Inventory $o_t$")
        ax_row[1].set_ylabel("Action")
        ax_row[1].set_yticks(actions, labels=[rf"$a_{i}$" for i in actions])
    
    axs[-1][0].set_xlabel(r"Step $t$")
    axs[-1][0].set_xticks([0, 30, 60, 90, 120, 150, 180])
    axs[-1][1].set_xlabel(r"Step $t$")        

    fig.suptitle(f'Inventories and actions in episodes {episode_ids}', y=1.05)    
    fig.tight_layout()

    file_name = f'training_observation_action_histories_{date_id}.png' if date_id else 'training_observation_action_histories.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_first_observation_values(training_results_dict, run_ids='all', mean=True, std=False, 
                                  figsize=None, line=None, date_id=None,
                                  save=False, show=True):
    """
    Plot first observation values v^*_0(n) across training episodes for 
    selected runs, optionally with mean and standard deviation bands.

    Args:
        training_results_dict (dict): Training results dictionary keyed by run id.
        run_ids (str, int, list of int, or empty list): Which runs to plot.
            'all' plots all runs, [] plots only mean/std. Defaults to 'all'.
        mean (bool): Whether to overlay the mean across runs. Defaults to True.
        std (bool): Whether to overlay +/- one std band. Defaults to False.
        figsize (tuple or None): Figure size (width, height). Defaults to None.
        line (float or None): Horizontal reference line value. Defaults to None.
        date_id (str or None): Date identifier appended to saved file name.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    plt.figure(figsize=figsize)      
    
    if run_ids == []: 
        means = np.mean([run['first_obs_values'] for run in training_results_dict.values()], axis=0)
        stds = np.std([run['first_obs_values'] for run in training_results_dict.values()], axis=0)
        plt.plot(means, label=r"mean first observation value $\bar v_0^*$", ls='-', lw=1.5, color='b')
        plt.fill_between(range(len(means)), means + stds, means - stds, alpha=0.2, color='b',
                        label=r"$\pm$ one standard deviation $\sigma_{v_0^*}$")
        if line is not None:
            plt.plot([line for _ in range(len(means))], ls=':', lw=1.2, color='black') 
        plt.legend(loc='best')
        plt.title(r'Average first observation value in training')
    else: 
        if type(run_ids) == str and run_ids == 'all':
            run_ids = training_results_dict.keys()
        elif type(run_ids) == int:
            run_ids = [run_ids]
    
        for run_id in run_ids:
            plt.plot(training_results_dict[run_id]["first_obs_values"], 
                    label=f"Run {run_id+1}", lw=1)          
        if line is not None:
            plt.plot([line for _ in range(len(training_results_dict[0]['first_obs_values']))], 
                     ls=':', lw=1.2, color='black')
        if mean:
            means = np.mean([run['first_obs_values'] for run in training_results_dict.values()], axis=0)
            plt.plot(means, label=r"mean $v_0^*(n)$", ls='--', lw=1.5, color='black')
        if mean and std:
            stds = np.std([run['first_obs_values'] for run in training_results_dict.values()], axis=0)
            plt.fill_between(range(len(means)), means + stds, means - stds, 
                            label=r"$\pm\sigma_{v_0^*(n)}$", alpha=0.2, color='black')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [*range(7), 10, 7, 8, 9, 11] if figsize[1] < 4 else [*(range(5)),10,*range(5,10),11]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                   loc='upper right', ncols=3 if figsize[1] < 4 else 2, fontsize=9)        
        plt.title('First observation values in training')    

    plt.xlabel(r'episode $n$', fontsize=11)
    plt.ylabel(r'$v_0^*(n)$', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)   
    plt.tight_layout()

    file_name = f'training_first_observation_values_{date_id}.png' if date_id else 'training_first_observation_values.png'
    if save:
        plt.savefig(FIGURES_DIR + file_name)
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_reward_vs_first_obs_value(training_results_dict, episode_window=(-500, -1), 
                                   figsize=None, line=None, date_id=None,
                                   save=False, show=True):
    """
    Plot average reward and average first observation value side by side 
    for a given episode window, with +/- one standard deviation bands.

    Args:
        training_results_dict (dict): Training results dictionary keyed by run id.
        episode_window (tuple of int): (start, end) episode indices for the 
            plotting window. Supports negative indexing. Defaults to (-500, -1).
        figsize (tuple or None): Figure size (width, height). Defaults to None.
        line (float or None): Horizontal reference line value. Defaults to None.
        date_id (str or None): Date identifier appended to saved file name.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    n_episodes = len(list(training_results_dict.values())[0]['rewards'])
    start = episode_window[0] if episode_window[0] >= 0 else n_episodes + episode_window[0]
    end = episode_window[1] if episode_window[1] >= 0 else n_episodes + episode_window[1]
    domain = range(start, end)
    # rewards
    means = np.mean([run['rewards'][start:end] for run in training_results_dict.values()], axis=0)
    stds_val = np.std([run['rewards'][start:end] for run in training_results_dict.values()], axis=0)
    axs[0].plot(domain, means, ls='-', lw=1.5, color='b', label=r"mean reward")
    axs[0].fill_between(domain, means + stds_val, means - stds_val, alpha=0.2, color='b',
                        label=r"$\pm$ $\sigma_{reward}$")
    if line is not None:
        axs[0].plot(domain, [line for _ in domain], ls=':', lw=1.2, color='black') 
    axs[0].set_ylim([-0.22, 0.05])    
    axs[0].set_ylabel(r'reward', fontsize=11)
    axs[0].set_xlabel(r'episode $n$', fontsize=11)
    axs[0].legend(loc='lower right', fontsize=9)
    axs[0].set_title(r'Average reward in training', fontsize=11)

    # first observation values
    means = np.mean([run['first_obs_values'][start:end] for run in training_results_dict.values()], axis=0)
    stds_val = np.std([run['first_obs_values'][start:end] for run in training_results_dict.values()], axis=0)    
    axs[1].plot(domain, means, ls='-', lw=1.5, color='b', label=r"mean $v_0^*(n)$")
    axs[1].fill_between(domain, means + stds_val, means - stds_val, alpha=0.2, color='b',
                        label=r"$\pm$ $\sigma_{v_0^*(n)}$")
    if line is not None:
        axs[1].plot(domain, [line for _ in domain], ls=':', lw=1.2, color='black') 
    axs[1].set_ylim([-0.10, -0.070])    
    axs[1].legend(loc='lower right', fontsize=9)
    axs[1].set_title(r'Average $v_0^*(n)$ in training', fontsize=11)
    axs[1].set_ylabel(r'$v_0^*(n)$', fontsize=11)
    axs[1].set_xlabel(r'episode $n$', fontsize=11)

    fig.tight_layout()

    file_name = f'training_reward_vs_first_observation_value_{date_id}.png' if date_id else 'training_reward_vs_first_observation_value.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


# --------------------------------------------------------------------------
# Testing plots
# --------------------------------------------------------------------------

def plot_test_run_results(test_results_dict, run_id, episode_id=-1, ma_window=0, figsize=None,
                          date_id=None, save=False, show=True):
    """
    Plot test results for a single run with rewards, terminal inventories, 
    actions, and inventories in a 2x2 subplot grid.

    Args:
        test_results_dict (dict): Test results dictionary keyed by run id.
        run_id (int): Run index to plot.
        episode_id (int or None): Episode index for actions/inventories. 
            Use None to overlay all episodes. Defaults to -1 (last episode).
        ma_window (int): Window size for simple moving average overlay. 
            Defaults to 0 (no SMA).
        figsize (tuple or None): Figure size (width, height). Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    test_run = test_results_dict[run_id]
    names = ["rewards", "terminal_inventories", "actions", "inventories"]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=figsize)
    for ax, id in zip(axes.flat, range(4)):
        ax.set_title(names[id] + f' in run {run_id}' if id < 2 else names[id] + (
                        " in episode " + str(episode_id) if episode_id != -1 else " in last episode")
        )
        if episode_id is not None:
            ax.plot(test_run[names[id]] if id < 2 else test_run[names[id]][episode_id])
        else:
            ax.plot(test_run[names[id]]) if id < 2 else [ax.plot(x) for x in test_run[names[id]]]

        if ma_window > 0:
            ax.plot(
                pd.Series(test_run[names[id]]).rolling(ma_window).mean()
                if id < 2 else
                pd.Series(test_run[names[id]][episode_id]).rolling(ma_window).mean(), 
                label="SMA {}".format(ma_window)
            )            
        ax.set_xlabel("Episodes" if id < 2 else "Steps")
    fig.tight_layout()

    file_name = f'test_run_{run_id}_results_{date_id}.png' if date_id else f'test_run_{run_id}_results.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_test_inventory_trajectories(test_results_dict, runs=None, n_steps=181,
                                     figsize=(10, 14), date_id=None, save=False, show=True):
    """
    Plot inventory trajectories for all episodes across multiple test runs.
    Each run is displayed in its own subplot in a grid layout.

    Args:
        test_results_dict (dict): Test results dictionary keyed by run id.
        runs (list of int or None): Run indices to plot. Defaults to None 
            (all runs).
        n_steps (int): Number of time steps per episode. Defaults to 181.
        figsize (tuple): Figure size (width, height). Defaults to (10, 14).
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    if runs is None:
        runs = list(test_results_dict.keys())
    
    n_runs = len(runs)
    n_cols = 2
    n_rows = (n_runs + 1) // n_cols
    domain = range(-5, n_steps + 5)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True)
    for ax, id_x in zip(axs.flatten(), runs):
        for episode in test_results_dict[id_x]['inventories']:
            ax.plot(episode)
        for value in [-50, 0, 50]:
            ax.plot(domain, [value for _ in domain], c='black', ls=':', lw=1.5)
        ax.set_xlim(-5, n_steps + 5)
        ax.set_ylim(-150, 750)
        ax.set_title(f'Inventories in test run {id_x+1}')
        if id_x % 2 == 0:
            ax.set_ylabel(r'Inventory $i_t$')
        if id_x in runs[-2:]:
            ax.set_xlabel(r'Step $t$')
    fig.tight_layout()

    file_name = f'test_inventory_trajectories_{date_id}.png' if date_id else 'test_inventory_trajectories.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_test_action_trajectories(test_results_dict, runs=None, n_steps=181,
                                  figsize=(10, 14), date_id=None, save=False, show=True):
    """
    Plot action trajectories for all episodes across multiple test runs.
    Each run is displayed in its own subplot in a grid layout.

    Args:
        test_results_dict (dict): Test results dictionary keyed by run id.
        runs (list of int or None): Run indices to plot. Defaults to None 
            (all runs).
        n_steps (int): Number of time steps per episode. Defaults to 181.
        figsize (tuple): Figure size (width, height). Defaults to (10, 14).
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    if runs is None:
        runs = list(test_results_dict.keys())

    n_runs = len(runs)
    n_cols = 2
    n_rows = (n_runs + 1) // n_cols
    domain = range(-5, n_steps + 5)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=True)
    for ax, id_x in zip(axs.flatten(), runs):
        for episode in test_results_dict[id_x]['actions']:
            ax.plot(episode)
        ax.set_xlim(-5, n_steps + 5)
        ax.set_title(f'Actions in test run {id_x+1}')
        if id_x % 2 == 0:
            ax.set_ylabel(r'Action $a_t$')
        if id_x in runs[-2:]:
            ax.set_xlabel(r'Step $t$')
    fig.tight_layout()

    file_name = f'test_action_trajectories_{date_id}.png' if date_id else 'test_action_trajectories.png'
    if save:
        fig.savefig(FIGURES_DIR + file_name, bbox_inches='tight')
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


def plot_test_rewards_boxplot(test_results_dict, baseline_results_dict=None, date_id=None,
                              figsize=(9, 4), save=False, show=True):
    """
    Box-plot of rewards for all test runs, optionally including a baseline 
    comparison as the last box.

    Args:
        test_results_dict (dict): Test results dictionary keyed by run id.
        baseline_results_dict (dict or None): Baseline results dictionary. 
            If provided, appended as the last boxplot entry. Defaults to None.
        date_id (str or None): Date identifier appended to saved file name.
        figsize (tuple): Figure size (width, height). Defaults to (9, 4).
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    all_rewards = [run['rewards'] for run in test_results_dict.values()]
    labels = ['Run {}'.format(i+1) for i in range(len(all_rewards))]
    if baseline_results_dict is not None:
        all_rewards.append(baseline_results_dict['rewards'])
        labels.append('Baseline')

    plt.figure(figsize=figsize)
    bp = plt.boxplot(all_rewards, showmeans=True, meanline=True, patch_artist=True, 
                        boxprops=dict(facecolor="white"))
    plt.plot([i for i in range(len(all_rewards)+2)], [0 for _ in range(len(all_rewards)+2)], 
             color='grey', ls=':', lw=1)
    plt.legend([bp['means'][0], bp['medians'][0], bp['boxes'][0], bp['whiskers'][0]], 
                  ['mean', 'median', 'IQR', r'$\pm$1.5 IQR'], loc='lower right', fontsize=9, ncol=2)

    plt.xlim(0.5, len(all_rewards)+0.5)
    plt.xticks(range(1, len(all_rewards)+1), labels)
    plt.ylabel('Reward')
    plt.tight_layout()

    file_name = f'test_runs_reward_boxplots_{date_id}.png' if date_id else 'test_runs_reward_boxplots.png'
    if save:
        plt.savefig(FIGURES_DIR + file_name)
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()


# --------------------------------------------------------------------------
# Confidence interval plots
# --------------------------------------------------------------------------

def plot_confidence_intervals(conf_intervals, key='rewards', date_id=None,
                              figsize=(5.5, 4.125), save=False, show=True):
    """
    Plot confidence intervals as horizontal error bars for each run.

    Args:
        conf_intervals (dict): Dictionary mapping run id to (lower, upper) 
            confidence interval bounds.
        key (str): Name of the metric (used in file name). Defaults to 'rewards'.
        date_id (str or None): Date identifier appended to saved file name.
        figsize (tuple): Figure size (width, height). Defaults to (5.5, 4.125).
        save (bool): Whether to save the plot. Defaults to False.
        show (bool): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    for run, (lower, upper) in conf_intervals.items():
        plt.plot((lower, upper), (run, run), 'b|-')
        plt.plot((lower+upper)/2, run, 'bo')
    plt.yticks(list(conf_intervals.keys()), fontsize=11,
               labels=[f'Run {run+1}' for run in conf_intervals.keys()])
    plt.xticks(fontsize=11) 
    plt.tight_layout()

    file_name = f'confidence_intervals_{key}_{date_id}.png' if date_id else f'confidence_intervals_{key}.png'
    if save:
        plt.savefig(FIGURES_DIR + file_name)
        print(f"Plot saved under '{FIGURES_DIR + file_name}'")
    if show:
        plt.show()
    else:
        plt.close()
