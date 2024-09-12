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


def save_results(results: Any, file_name: str):
    assert(type(file_name) == str), "file_name must be a string."

    date_time = datetime.now().strftime("_%d-%m-%Y_%H:%M:%S")
    file_name += date_time
    try:
        with open("../results/" + file_name + ".pkl", "xb") as fout:
            pickle.dump(results, fout)
        print(f"Passed results saved as: '{file_name}'")
    except FileExistsError: 
        warnings.warn(
            f"File with name '{file_name}' already exists. \\ Passed result were NOT saved."
        )

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

def scatter_plot_positions(results):
    plt.scatter(results[4], results[2])
    plt.ylabel("End position (>= 0.5 is a success)")
    plt.xlabel("Start position")
    plt.title("Start vs. end positions")
    plt.show()  


def save_plots(results, method, plot_id=None, subplot=True, state='partial', window=100):
    assert plot_id != None, "Provide plot identification."
    if state not in ['partial', 'full']:
        raise ValueError("state must be in {partial, full}.")

    if method not in ['qlearning', 'reinforce']:
        raise ValueError("method must be in {qlearning, reinforce}.")

    path = '../figures/mountainCar/' + state + '_state/'
    ylabels = ['Rewards', 'Loss', 'End position', 'Steps']
    order = [0, 2, 1, 3]
    # order = range(4)
    
    if subplot:
        plt_name = (method + '_' 
                    + str(plot_id) + '_all.png')
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
        for ax, id, count in zip(axes.flat, order, range(4)):
            ax.set_title(ylabels[id])
            ax.plot(results[id])
            ax.plot(moving_average(results[id], window))
            if count > 1:
                ax.set_xlabel('Episodes')
        fig.tight_layout()
        plt.savefig(path + plt_name)
        plt.close()     
    
    else:
        for id, label in zip(range(4), ylabels):
            plt_name = (method + '_' + plot_id + '_' 
                        + str(id) + '_' + label + '.png')
            plt.plot(results[id])
            if window != None:
                plt.plot(moving_average(results[id], window))
            plt.xlabel('Episode')
            plt.ylabel(label)
            plt.savefig(path + plt_name)
            plt.close()


def make_reproducable(base_seed=0, numpy_seed=0, torch_seed=0):
    seed(base_seed)
    np.random.seed(numpy_seed)
    manual_seed(torch_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

 
def exponential_decay(factor=1, min_value=0., wait=0):
    assert 0 < factor <= 1, "multiplicative decay :factor: in :mode: 'exponential' \
                            must be in intervall [0, 1]."
    assert 0 <= wait, ":wait: epochs must be greater or equal to zero."
    
    # return multiplicative decay factor
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            max(factor ** epoch, min_value))
    )
     
def linear_decay(epochs, start, end=0., wait=0, steps=None):
    assert epochs > 0, "Choose positive integer value as total decay epochs."
    assert 0 <= wait <= epochs, "Choose integer value between 0 and decay epochs as wait epochs."

    if steps == None:
        steps = epochs - wait
    else:
        assert steps > 0, "Choose positive integer value as number of steps to decay over."
    
    epochs -= wait 
    frac = end / start    
    step_length = np.ceil(epochs / (steps+1))
    
    # return factor to decay :start: to :end: linearly in :steps: over :epochs: - :wait:
    # assuming :epoch: is counted from 0 to :epochs:-1
    return lambda epoch: (
        1 if (
            epoch < wait
        ) else ( 
            1 - (1 - frac) * min( ((epoch - wait) // step_length) / steps, 1))
        )    

def mixed_linear_decay(epochs, switch, schedule_1=None, schedule_2=None):
    assert switch <= epochs, "Must switch schedule before end of epochs"
    return lambda epoch: (
        schedule_1(epoch) if epoch < switch else (
            schedule_2(epoch-switch)*schedule_1(switch)
        )
    )
    


    
