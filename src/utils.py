from datetime import datetime
import pickle
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


""" 
TODO: create methods
    - plot_average_results
    - save_average_results_plots
"""


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


def plot_run_results(training_results, run_id=-1, subplot=True, index=None, ma_window=50, point=False, figsize=None):
    results_run = training_results[run_id]
    names = results_run.keys()[0:4] # rewards, losses, cash, terminal_inventory
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    for ax, y_label, count in zip(axes.flat, names, range(4)):
        ax.plot(results_run[y_label])
        ma = ax.plot(pd.Series(results_run[id]).rolling(ma_window).mean(), label="SMA {}".format(ma_window))
        if count ==1:
            fig.legend(ma, loc='outside upper right')
        if count > 1:
            ax.set_xlabel('Episodes', size=11)

    fig.suptitle('Training results for run {}'.format(run_id), size=15)
    fig.tight_layout()
    plt.show()


def plot_observation_action_history(observation_history, action_history, episode_id, 
                                    n_actions=3, figsize=(6,3)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(observation_history)
    ax[0].set_xlabel(f"Observation history in episode {episode_id}")
    ax[0].legend(["Time pct", "Inventory pct"], loc="best")

    ax[1].plot(action_history)
    ax[1].set_xlabel(f"Action history in episode {episode_id}")
    actions = [*range(n_actions)]
    ax[1].set_yticks(actions, labels=[f"A{i}" for i in actions])
    fig.tight_layout()
    plt.show()    


def plot_mean_results(training_results, title=None, ma_window=50, figsize=None):
    # calculate mean and std over runs
    results_keys = training_results[-1].keys()[0:4] # rewards, losses, cash, terminal_inventory
    results_array = np.array([
        training_results[run][key] 
        for run in training_results.keys() 
        for key in results_keys
    ])
    means = np.mean(results_array, axis=0)
    stds = np.std(results_array, axis=0)
    
    # plot mean and std
    y_labels = ['reward', 'loss', 'cash', 'terminal inventory'] # keys differ from names
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, layout='constrained', figsize=figsize)
    for ax, id in zip(axes.flat, range(4)):
        ax.plot(means[id], color="b", label="mean over runs") 
        ax.plot(pd.Series(means[id]).rolling(ma_window).mean())
        ax.fill_between(range(len(means[id])),
                        means[id] - 1 * stds[id],
                        means[id] + 1 * stds[id],
                        color='b', alpha=0.2, label='+/- one standard deviation')
        ax.set_ylabel('mean ' + y_labels[id])
        if id > 1:
            ax.set_xlabel('episodes')
        #ax.legend(loc="best") # legend in each subplot
        if id==0:
            fig.legend(bbox_to_anchor=(0.18, 1.02, 1., .102), loc='lower left',
                      ncols=2, borderaxespad=-0.2)
    fig.suptitle(title, y=1.127)
    plt.show()

def save_mean_results_plots(training_results, date_time_id, file_name, file_path='../figures', 
                            title=False, ma_window=50, figsize=(5.5,4.125), show=False):
    # calculate mean and std over runs
    results_keys = training_results[-1].keys()[0:4] # rewards, losses, cash, terminal_inventory
    results_array = np.array([
        training_results[run][key] 
        for run in training_results.keys() 
        for key in results_keys
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
        plt.savefig(file_path + file_name + 'mean_' + y_label + '_' + date_time_id + '.png')
        if show:
            plt.show()
        else:
            plt.close()


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


def plot_confidence_intervals(conf_intervals, key='rewards', save=False, date_time_id=None):
    if save:
        assert date_time_id != None, "Provide date_time_id to save the plot."
    for run, (lower, upper) in conf_intervals.items():
        plt.plot((lower, upper), (run, run), 'b|-')
        plt.plot((lower+upper)/2, run, 'bo')
    plt.yticks(conf_intervals.keys(), fontsize=11,
               labels=[f'Run {run+1}' for run in conf_intervals.keys()])
    plt.xticks(fontsize=11) 
    plt.figsize=(5.5, 4.125)
    plt.tight_layout()
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
# other utility functionalities
#--------------------------------------------------

def save_results(results: Any, file_name: str):
    """
    Save results to a pickle file in the results directory. Returns
    a unique date_time_id string to use for saving corresponding plots.
    
    Args:
        results: results to save, can be any type serializable by pickle
        file_name: name of the file to save the results to

    Returns:
        date_time_id: string with date and time of saving and a unique id
    """
    assert(type(file_name) == str), "file_name must be a string."

    date_time = datetime.now().strftime("_%Y%m%d")
    id = ord('A')
    exists = True
    while exists:
        date_time_id = date_time + '_' + chr(id)
        file_path = '../results/' + file_name + date_time_id + '.pkl'
        try:
            with open(file_path, "xb") as fout:
                pickle.dump(results, fout)
            print(f"Passed results saved under: '{file_path}'")
            exists = False
        except FileExistsError: 
            warnings.warn(
                f"File '{file_path}' already exists. Setting new date_time_id."
            )
            id += 1
    return date_time_id

def generate_prime_seeds(m, shuffle=False):
    """
    Returns an array of m primes, where 1 <= m <= 100.
    The main part of this function's body is the function `primesfrom2to` from
    https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n?noredirect=1&lq=1,
    which for an integer input n>=6, returns an array of primes, 2 <= p < n
    """
    assert(m <= 100), ValueError("Maximum number of seeds is 100.")

    n = 542 # number of primes below 543 is 100
    sieve = np.ones(n//3 + (n%6==2), dtype=bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    primes = np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]
    return primes.tolist() if not shuffle else np.random.permutation(primes).tolist()        

def make_reproducable(base_seed=0, numpy_seed=0, torch_seed=0):
    seed(base_seed)
    np.random.seed(numpy_seed)
    manual_seed(torch_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
