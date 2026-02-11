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
# plotting functionalities (moved to plotting_utils.py)
#--------------------------------------------------------------------------
# NOTE: All plotting functions have been moved to src/plotting_utils.py.
# The following utility functions remain here as they are used by other
# modules (e.g. livelossplot callback, training scripts).


def custom_after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Set properties after subplot creation from livelossplot.
    Custom variation of default function to change legend location.

    Args:
        fig: matplotlib Figure
        group_name: name of metrics group (eg. Accuracy, Recall)
        x_label: label of x axis (eg. epoch, iteration, batch)        
    
    Returns:
        None, modifies the given axis in-place.
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
        raise ValueError(f"result_type must be one of {list(file_name_map.keys())}, got '{results_type}'")
    
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
        raise ValueError(f"result_type must be one of {list(file_name_map.keys())}, got '{results_type}'")
    
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
    
