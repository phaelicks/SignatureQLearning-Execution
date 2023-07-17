import matplotlib.pyplot as plt
import numpy as np
from random import seed
from torch import manual_seed
from torch.backends import cudnn

def moving_average(seq, window):
    seq = np.array(seq)
    moving_avg = []
    moving_avg.append(seq[0])
    for i in range(len(seq)):
        j = i + 1 - window
        mean = seq[max(0 , j): i+1].mean()
        moving_avg.append(mean)
    return moving_avg


def plot_results(results, subplot=True, index=None, window=100, point=False):
    
    names = ['Rewards', 'Loss', 'End position', 'Steps']
    order = [0, 2, 1, 3] 
    #order = range(4)

    if subplot:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
        for ax, id, count in zip(axes.flat, order, range(4)):
            ax.set_title(names[id])
            ax.plot(results[id])
            ax.plot(moving_average(results[id], window))
            if count > 1:
                ax.set_xlabel('Episodes')
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
    assert epochs > 0, "number of total :epochs: to decay over in :mode: 'linear' \
                        must be an integer greater than zero."
    assert 0 <= wait <= epochs, ":wait: epochs must be an integer between 0 and :epochs:"

    if steps == None:
        steps = epochs - wait
    else:
        assert steps > 0, "number of :steps: to decay over in :mode: 'linear' \
                            must be an integer greater than zero."
    
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
    


    
