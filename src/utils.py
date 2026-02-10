from datetime import datetime
import pickle
from pathlib import Path
from random import seed
import tqdm.notebook as tqdm
from typing import Any, Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch import manual_seed
from torch.backends import cudnn

#--------------------------------------------------------------------------
# plotting functionalities
#--------------------------------------------------------------------------


def custom_after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Set properties after subplot creation from livelossplot.
    Custom variation of default function to change legend location.
    Args:
        fig: matplotlib Figure
        group_name: name of metrics group (eg. Accuracy, Recall)
        x_label: label of x axis (eg. epoch, iteration, batch)        
    """
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc='best')


def moving_average(seq, window):
    seq = np.array(seq)
    moving_avg = []
    for i in range(len(seq)):
        j = i + 1 - window
        mean = seq[max(0 , j): i+1].mean()
        moving_avg.append(mean)
    return moving_avg


def plot_training_run_results(training_results, run_id, ma_window=50, figsize=None):
    results_run = training_results[run_id]
    keys = list(results_run.keys())[0:4] # rewards, losses, cash, terminal_inventory
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=figsize)
    for ax, y_label, count in zip(axes.flat, keys, range(4)):
        ax.plot(results_run[y_label])
        ma = ax.plot(pd.Series(results_run[y_label]).rolling(ma_window).mean(), label="SMA {}".format(ma_window))
        ax.set_ylabel(y_label)
        if count ==1:
            fig.legend(handles=ma, loc='outside upper right')
        if count > 1:
            ax.set_xlabel('Episodes')

    fig.suptitle('Training results for run {}'.format(run_id), size=15)
    fig.tight_layout()
    plt.show()


def plot_mean_training_results(training_results, title=None, ma_window=50, figsize=None):
    # calculate mean and std over runs
    plot_keys = list(next(iter(training_results.values())).keys())[0:4] # rewards, losses, cash, terminal_inventory
    results_array = np.array([
        [run_results[key] for key in plot_keys] 
        for run_results in training_results.values()
    ])
    means = np.mean(results_array, axis=0)
    stds = np.std(results_array, axis=0)
    
    # plot mean and std
    y_labels = ['reward', 'loss', 'cash', 'terminal inventory'] # y_labels differ from plot_keys
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
        #ax.legend(loc="best") # legend in each subplot
        if id==0:
            fig.legend(bbox_to_anchor=(0.18, 1.02, 1., .102), loc='lower left',
                      ncols=2, borderaxespad=-0.2)
    fig.suptitle(title, y=1.127)
    plt.show()


def save_mean_training_results_plots(training_results, date_time_id, file_name, file_path='../figures/', 
                                     title=False, ma_window=50, figsize=(5.5,4.125), show=False):
    # calculate mean and std over runs
    plot_keys = list(next(iter(training_results.values())).keys())[0:4] # rewards, losses, cash, terminal_inventory
    results_array = np.array([
        [run_results[key] for key in plot_keys] 
        for run_results in training_results.values()
    ])
    means = np.mean(results_array, axis=0)
    stds = np.std(results_array, axis=0)

    y_labels = ['reward', 'loss', 'cash', 'terminal inventory'] # keys differ from names
    for y_label, id in zip(y_labels, range(4)):
        plt.figure(figsize=figsize)        
        plt.plot(means[id], color="b", label='mean {}'.format(y_label))
        plt.plot(pd.Series(means[id]).rolling(ma_window).mean())
        # create std intervals to fill
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
        if title:
            plt.title('Average {} over training runs'.format(y_label), fontsize=10)
        else: 
            plt.title(' ', fontsize=12)
        plt.tight_layout()
        plt.savefig(file_path + file_name + '_mean_' + y_label.replace(' ', '_') + '_' + date_time_id + '.png')
        if show:
            plt.show()
        else:
            plt.close()


def plot_inventory_action_histories(training_results_dict, run_ids=0, episode_ids=-1, n_actions=3,
                                    figsize=None, save=False, date_time_id=None):
    """
    Plots observation and action histories for given run_ids and episode_ids.

    Args:
        training_results_dict: dict with training results
        run_ids: list of run_ids or single int run_id as int
        episode_ids: list of episode_id or single int episode_id
        n_actions: total number of actions
        figsize: figure size
        save: boolean flag if plot should be saved
        date_time_id: string to append to file name for saving, 
                      should be the date_time_id from provided training_results_dict

    Returns:
        A plot with inventory and action histories for all runs in run_ids for each
        episode in episode_ids. Each subplot shows inventory and action histories, 
        respectively, for a single episode_id and all run_ids.
    """
    if type(run_ids) == str and run_ids == 'all': 
        run_ids = list(training_results_dict.keys())
    elif type(run_ids) == int:
        run_ids = [run_ids]
    if type(episode_ids) == int:
        episode_ids = [episode_ids]
    
    plot_rows = len(episode_ids)
    max_episodes = len(training_results_dict[0]["observations"])
    if figsize == None:
        figsize = (9, 2.5 * plot_rows)
    for i in range(plot_rows):
        episode_ids[i] = episode_ids[i] if episode_ids[i] >= 0 else max_episodes + episode_ids[i]

    fig, axs = plt.subplots(plot_rows, 2, sharex=True, figsize=figsize, squeeze=False)
    for episode_id, ax_row in zip(episode_ids, axs):
        for run_id in run_ids:
            # extract observations and actions
            observation_history = list(training_results_dict[run_id]["observations"][episode_id])
            action_history = training_results_dict[run_id]["actions"][episode_id]
            # plot
            ax_row[0].plot([inventory for (_, inventory) in observation_history])
            ax_row[0].set_title(f'Episode {episode_id+1}', fontsize=10)
            ax_row[1].plot(action_history, label='Run {}'.format(run_id+1))
            ax_row[1].set_title(f'Episode {episode_id+1}', fontsize=10)

        if episode_id == episode_ids[0]: # legend only once
            fig.legend(bbox_to_anchor=(0.22, 0.94, 1., .102), loc='center left',
                        ncols=5, borderaxespad=-0.2)

    actions = [*range(n_actions)]
    for ax_row in axs:
        ax_row[0].set_ylabel(r"Inventory $o_t$")
        ax_row[1].set_ylabel("Action")
        ax_row[1].set_yticks(actions, labels=[rf"$a_{i}$" for i in actions])
    
    axs[-1][0].set_xlabel(r"Step $t$")
    axs[-1][0].set_xticks([0, 30, 60, 90, 120, 150, 180]) # one axis seems enough
    axs[-1][1].set_xlabel(r"Step $t$")        

    fig.suptitle(f'Inventories and actions in episodes {episode_ids}', y=1.05)    
    fig.tight_layout()
    if save:
        fig.savefig('../figures/custom_execution_observation_action_histories_{}.png'.format(
            date_time_id
        ), bbox_inches='tight')    
    plt.show()    
    

def plot_first_observation_values(training_results_dict, run_ids='all', mean=True, std=False, figsize=None,
                                  line=None, save=False, date_time_id=None):
    plt.figure(figsize=figsize)      
    
    # plot only mean and std
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
    
    # plot specified runs
    else: 
        if type(run_ids) == str and run_ids == 'all': # all runs
            run_ids = training_results_dict.keys()
        elif type(run_ids) == int: # single run
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
                   loc='upper right', ncols=3 if figsize[1] <4 else 2, fontsize=9)        
        plt.title('First observation values in training')    

    plt.xlabel(r'episode $n$', fontsize=11)
    plt.ylabel(r'$v_0^*(n)$', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)   
    plt.tight_layout()
    if save:
        file_path = '../figures/'
        file_path += 'first_observation_values' if run_ids is not [] else 'mean_first_observation_value'
        plt.savefig(file_path + '_' + date_time_id + '.png')
    plt.show()


def plot_reward_vs_first_obs_value(training_results_dict, episode_window=(-500,-1), figsize=None,
                                   line=None, save=False, date_time_id=None):
    fig, axs =  plt.subplots(1, 2, figsize=figsize)
    n_episodes = len(list(training_results_dict.values())[0]['rewards'])
    start = episode_window[0] if episode_window[0] >= 0 else n_episodes + episode_window[0]
    end = episode_window[1] if episode_window[1] >= 0 else n_episodes + episode_window[1]
    domain = range(start, end)
    # rewards
    means = np.mean([run['rewards'][start:end] for run in training_results_dict.values()], axis=0)
    stds = np.std([run['rewards'][start:end] for run in training_results_dict.values()], axis=0)
    axs[0].plot(domain, means, ls='-', lw=1.5, color='b', label=r"mean reward")
    axs[0].fill_between(domain, means + stds, means - stds, alpha=0.2, color='b',
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
    stds = np.std([run['first_obs_values'][start:end] for run in training_results_dict.values()], axis=0)    
    axs[1].plot(domain, means, ls='-', lw=1.5, color='b', label=r"mean $v_0^*(n)$")
    axs[1].fill_between(domain, means + stds, means - stds, alpha=0.2, color='b',
                        label=r"$\pm$ $\sigma_{v_0^*(n)}$")
    if line is not None:
        axs[1].plot(domain, [line for _ in domain], ls=':', lw=1.2, color='black') 
    axs[1].set_ylim([-0.10, -0.070])    
    axs[1].legend(loc='lower right', fontsize=9)
    axs[1].set_title(r'Average $v_0^*(n)$ in training', fontsize=11)
    axs[1].set_ylabel(r'$v_0^*(n)$', fontsize=11)
    axs[1].set_xlabel(r'episode $n$', fontsize=11)
    #axs[1].set_xticklabels([str(i) for i in range(start, end, 100)])

    fig.tight_layout()
    if save:
        file_path = '../figures/reward_vs_first_observation_value'
        plt.savefig(file_path + '_' + date_time_id + '.png')
    plt.show()


def plot_test_run_results(test_results_dict, run_id, episode_id=-1, ma_window=0, figsize=None):
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
    plt.show()

def plot_baseline_results(baseline_results, episode_id=-1, ma_window=0, figsize=None):
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
    plt.show()

def save_baseline_trajectories(baseline_results, episode_ids=[-1], rho=50,
                               show=False, file_path=None):
    assert file_path != None, "Provide file_path to save the plots."
    assert len(episode_ids) <= 4, "Maximum of 4 episodes can be plotted."

    linestyles = ['-', '--', '-.', ':'][0:len(episode_ids)] # for different episodes
    plt.rcParams['text.usetex'] = False
    steps = len(baseline_results["inventories"][0])

    # actions
    plt.figure(figsize=(5.5, 4.125))
    for episode, style in zip(episode_ids, linestyles):
        plt.plot(baseline_results["actions"][episode], lw=2, ls=style)
    plt.xlabel("Step", fontsize=11)
    plt.xticks([0+ 20*i for i in range(10)], fontsize=11)
    plt.ylabel("Action", fontsize=11)
    plt.yticks([0, 1, 2], fontsize=11)
    plt.legend(["Episode one", "Episode two"], 
               loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(file_path + 'baseline_actions.png')
    plt.show() if show else plt.close()

    # terminal inventory
    plt.figure(figsize=(5.5, 4.125))
    for episode, style in zip(episode_ids, linestyles):
        plt.plot(baseline_results["inventories"][episode], lw=2, ls=style)
    plt.plot([rho//2 for _ in range(steps)], ls='dotted', color='black', lw=1.5)
    plt.plot([-rho//2 for _ in range(steps)], ls='dotted', color='black', lw=1.5)
    plt.xlabel("Step", fontsize=11)
    plt.xticks([0+ 20*i for i in range(steps//20+1)], fontsize=11)
    plt.ylabel("Inventory", fontsize=11)
    plt.legend(["Episode one", "Episode two", r"$\pm\,\rho\, /\, 2$"], 
               loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(file_path + 'baseline_inventories.png')
    plt.show() if show else plt.close()

#--------------------------------------------------------------------------
# confidence intervals for test results
#--------------------------------------------------------------------------

def compute_confidence_intervals(test_results, key='rewards', level=0.95):
    conf_intervals = {}
    mean_std_error = 0
    for run, results in test_results.items():
        mean = np.mean(results[key])
        standard_error = stats.sem(results[key])
        mean_std_error += standard_error/len(test_results)            
        confidence_interval = stats.norm.interval(level, loc=mean, scale=standard_error)
        conf_intervals[run] = confidence_interval
    return conf_intervals, mean_std_error


def plot_confidence_intervals(conf_intervals, key='rewards', save=False, date_time_id=None,
                              figsize=(5.5, 4.125)):
    if save:
        assert date_time_id != None, "Provide date_time_id to save the plot."
    for run, (lower, upper) in conf_intervals.items():
        plt.plot((lower, upper), (run, run), 'b|-')
        plt.plot((lower+upper)/2, run, 'bo')
    plt.yticks(list(conf_intervals.keys()), fontsize=11,
               labels=[f'Run {run+1}' for run in conf_intervals.keys()])
    plt.xticks(fontsize=11) 
    plt.figsize=figsize
    plt.tight_layout()
    if save:
        plt.savefig(f'../figures/confidence_intervals_{key}_{date_time_id}.png')
    plt.show()


#--------------------------------------------------------------------------
# hyperparameter decay schedules
#--------------------------------------------------------------------------

def create_decay_schedule(mode=None, start_value=None, **kwargs):
    if mode == None:
        return lambda step: 1
    elif mode == 'linear':
        return linear_decay(start_value=start_value, **kwargs)
    elif mode == 'exponential':
        return exponential_decay(**kwargs)
    else:
        raise ValueError('Decay mode must be None, linear or exponential.')

def exponential_decay(factor=1, end_value=0., wait=0):
    assert 0 < factor <= 1, "multiplicative decay factor in [0,1] in mode 'exponential'."
    assert 0 <= wait, "wait epochs must be non-negative integer."
    
    # return multiplicative decay factor
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            max(factor ** epoch, end_value))
    )
    
def linear_decay(epochs, start_value, end_value=0., wait=0, steps=None):
    assert epochs > 0, "Choose positive integer for total decay epochs."
    assert 0 <= wait <= epochs, "Choose integer value between 0 and decay epochs as wait epochs."

    if steps == None:
        steps = epochs - wait
    else:
        assert steps > 0, "Choose positive integer value as number of steps to decay over."
    
    epochs -= wait 
    frac = end_value / start_value    
    step_length = np.ceil(epochs / (steps+1))
    
    # return factor to decay :start: to :end: linearly in :steps: over :epochs: - :wait:
    # assuming :epoch: is counted from 0 to :epochs:-1
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            1 - (1 - frac) * min( ((epoch - wait) // step_length) / steps, 1))
        )    

#--------------------------------------------------
# results saving and loading functionalities
#--------------------------------------------------

def get_date_id():
    """
    Returns a unique date_id string to use for saving results and plots.
    The string contains the current date and a unique id to
    avoid overwriting existing files with the same date_id.

    Returns:
        date_id: string with current date and a unique id
    """
    warnings.resetwarnings() 

    date = datetime.now().strftime("%Y%m%d")
    id = ord('A')
    results_dir = Path('../results')
    
    while True:
        date_id = date + '_' + chr(id)
        # Check if any file ends with the date_id pattern
        pattern = f'*{date_id}.pkl'
        existing_files = list(results_dir.glob(pattern))
        
        if not existing_files:  # No files found with this date_id
            print('Current date_id: {}'.format(date_id))
            return date_id
        
        warnings.warn(
            f"Files with date_id '{date_id}' already exist. Setting new date_id."
        )
        id += 1    

def save_results(results: Any, date_id: str, results_type: str):
    """
    Save results to a pickle file in the results directory with a given date_id.
    If a file with the same type and date_id combination already exists,
    a warning is raised and no file is saved.
        
    Args:
        results: results to save, can be any type serializable by pickle
        date_id: unique date_id string (format: YYYYMMDD_X, e.g., '20250206_A')
        result_type: type of results ('baseline', 'training', or 'testing')

    Returns:
        date_id: the provided date_id if save was successful
        
    Raises:
        ValueError: if result_type is not recognized
    """
    assert isinstance(date_id, str), "date_id must be a string."
    assert isinstance(results_type, str), "result_type must be a string."
    
    # Map result types to file names
    file_name_map = {
        'baseline': 'execution_baseline_results',
        'training': 'execution_training_results',
        'testing': 'execution_test_results',
    }
    
    if results_type not in file_name_map:
        raise ValueError(f"result_type must be one of {list(file_name_map.keys())}, got '{result_type}'")
    
    file_name = file_name_map[results_type]
    file_path = f'../results/{file_name}_{date_id}.pkl'
    
    # Check if file already exists
    if Path(file_path).exists():
        warnings.warn(
            f"File '{file_path}' already exists. Results were NOT saved."
        )
        return date_id
    
    # Save results
    try:
        with open(file_path, 'wb') as fout:
            pickle.dump(results, fout)
        print(f"Results SAVED under: '{file_path}'")
        return date_id
    except Exception as e:
        warnings.warn(f"Error saving results: {e}")
        raise


def load_results(date_id: str, results_type: str):
    """
    Load results from a pickle file in the results directory with a given date_id.
    
    Args:
        result_type: type of results ('baseline', 'training', or 'testing')
        date_id: unique date_id string (format: YYYYMMDD_X, e.g., '20250206_A')

    Returns:
        For 'baseline': baseline_results_dict
        For 'training': dict with keys ['training_results', 'final_Q_functions', 
                                        'sig_params', 'training_params', 
                                        'env_params', 'training_seeds']
        For 'testing': dict with keys ['test_results_dict', 'test_seeds', 
                                       'checkpoint']
        
    Raises:
        ValueError: if result_type is not recognized
        FileNotFoundError: if the file does not exist
    """
    assert isinstance(date_id, str), "date_id must be a string."
    assert isinstance(results_type, str), "result_type must be a string."
    
    # Map result types to file names
    file_name_map = {
        'baseline': 'execution_baseline_results',
        'training': 'execution_training_results',
        'testing': 'execution_test_results',
    }
    
    if results_type not in file_name_map:
        raise ValueError(f"result_type must be one of {list(file_name_map.keys())}, got '{result_type}'")
    
    file_name = file_name_map[results_type]
    file_path = f'../results/{file_name}_{date_id}.pkl'
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    # Load results
    try:
        with open(file_path, 'rb') as fin:
            results = pickle.load(fin)
        print(f"Results of type '{results_type}' LOADED from: '{file_path}'") 
        return results
    except Exception as e:
        raise RuntimeError(f"Error loading results from '{file_path}': {e}") 
    

def unpack_training_results(training_data: dict) -> tuple:
    """
    Unpack training data dict into individual variables.
    
    Args:
        training_data: dict from load_results('training', date_id)
    
    Returns:
        tuple: (training_results_dict, final_Q_functions, sigq_params, 
                training_params, env_params, training_seeds, n_runs)
    """
    training_results_dict = training_data['training_results']
    final_Q_functions = training_data['final_Q_functions']
    sigq_params = training_data['sig_params']
    training_params = training_data['training_params']
    env_params = training_data['env_params']
    training_seeds = training_data['training_seeds']
    n_runs = len(training_results_dict)
    
    return (training_results_dict, final_Q_functions, sigq_params, 
            training_params, env_params, training_seeds, n_runs)


#--------------------------------------------------
# other utility functionalities
#--------------------------------------------------

def generate_prime_seeds(m, random=False):
    """
    Returns an array of m primes, where 1 <= m <= 100.
    The main part of this function's body is the function `primesfrom2to` from
    https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n?noredirect=1&lq=1,
    which for an integer input n>=6, returns an array of primes, 1 <= p < n
    """
    assert(m <= 100), ValueError("Maximum number of seeds is 100.")

    n = 542 # number of primes below 543 is 100
    sieve = np.ones(n//3 + (n%6==2), dtype=bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    primes = np.r_[1,2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]
    return primes.tolist()[0:m] if not random else np.random.permutation(primes).tolist()[0:m]        

def make_reproducable(base_seed=0, numpy_seed=0, torch_seed=0):
    seed(base_seed)
    np.random.seed(numpy_seed)
    manual_seed(torch_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
