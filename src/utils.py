from datetime import datetime
import pickle
from random import seed
import tqdm.notebook as tqdm
from typing import Any, Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
from torch import manual_seed
from torch.backends import cudnn


""" 
TODO: create methods
    - plot_average_results
    - save_average_results_plots
"""


#--------------------------------------------------
# plotting functionalities
#--------------------------------------------------


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


def plot_results(results, subplot=True, index=None, window=50, point=False, size=None):
    names = ['Reward', 'Loss', 'Cash', 'Terminal inventory']
    #order = [0, 2, 1, 3] 
    order = range(4)

    if subplot:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
        if size is not None:
            fig.set_size_inches(size)
        for ax, id, count in zip(axes.flat, order, range(4)):
            ax.set_title("{} per episode".format(names[id]), size=10)
            ax.plot(results[id])
            ax.plot(moving_average(results[id], window))
            if count > 1:
                ax.set_xlabel('Episodes', size=10)
        fig.tight_layout()
        plt.show()
 
    else: 
        if point:
            plt.plot(results[index], 'bo')
        else: 
            plt.plot(results[index])
        plt.plot(moving_average(results[index], window))
        plt.xlabel('Episode')
        plt.ylabel(names[index])
        plt.show()

#--------------------------------------------------
# hyperparameter decay schedules
#--------------------------------------------------

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
    assert(type(file_name) == str), "file_name must be a string."

    date_time = datetime.now().strftime("_%Y%m%d")
    id = ord('A')
    exists = True
    while exists:
        file_path = '../results/' + file_name + date_time + '_' + chr(id) + '.pkl'
        try:
            with open(file_path, "xb") as fout:
                pickle.dump(results, fout)
            print(f"Passed results saved under: '{file_path}'")
            exists = False
        except FileExistsError: 
            warnings.warn(
                f"File '{file_path}' already exists. Setting new identifier."
            )
            id += 1

def generate_prime_seeds(m):
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
    return primes.tolist()        

def make_reproducable(base_seed=0, numpy_seed=0, torch_seed=0):
    seed(base_seed)
    np.random.seed(numpy_seed)
    manual_seed(torch_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
