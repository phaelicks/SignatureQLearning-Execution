import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import tqdm
from copy import deepcopy
from typing import Optional, Dict, List
from math import floor
import warnings


def test(
        env, 
        policy, 
        episodes, 
        discount: float = 1.0,
        epsilon: float = 0.02, 
        window_length: Optional[int] = None,
        printing: bool = False
) -> Dict[str, List]: 
    reward_history = []
    inventory_history = []
    action_history = []
    cash_history = []

    # CHECK PROPERTIES
    assert (window_length == None) or (
        type(window_length) == int and window_length > 0
    ), "History window length must be a positive integer or None."
    
    # compute first_interval steps to solely observe
    try: 
        do_nothing_steps = (floor(env.observe_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("'env' admits no observation interval.")
    finally:
        do_nothing_steps = 0

    # TESTING
    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        episode_reward = 0
        episode_inventories = []
        episode_actions = []


        observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
        action = env.do_nothing_action_id # do nothing
        initial_tuple = np.hstack((observation, action))
        episode_actions.append(action)

        history = deque(maxlen=window_length)
        history.append(initial_tuple)

        initial_tuple_tensor = torch.tensor(
            [initial_tuple], requires_grad=False, dtype=torch.float
        )

        # initialize signature variable
        history_signature = policy.update_signature(initial_tuple_tensor.unsqueeze(0))
        last_tuple_tensor = initial_tuple_tensor

        # first step outside of loop
        done = False
        observation, reward, done, _ = env.step(action)
        observation = observation[:,0]

        # RUN EPISODE
        step_counter = 0
        while not done:
            if step_counter < do_nothing_steps:
                # agent only observes 
                action = 9 # next action is 'do nothing'
                new_tuple = np.hstack((observation, action))
                history.append(new_tuple)
                episode_actions.append(action)
                episode_inventories.append(0)

                observation, _, _, _ = env.step(action)
                observation = observation[:,0]
                step_counter += 1
                last_tuple_tensor = torch.tensor(
                    [new_tuple], requires_grad=False, dtype=torch.float
                )
                continue

            if step_counter == do_nothing_steps:
                # calculate first signature for observed history so far
                history_signature = policy.update_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0)
                )                
            
            # create Q values and select action
            Q = policy.create_Q_values(history_signature, last_tuple_tensor, observation)
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            episode_actions.append(action)

            # add (o,a) tuple to history
            next_tuple = np.hstack((observation, action))
            next_tuple_tensor = torch.tensor(
                [next_tuple], requires_grad=False, dtype=torch.float
            )                     

            # update history and signature
            history.append(next_tuple) # pops left if maxlen != None               
            history_signature = policy.update_signature(
                torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0)
            )
            # TODO: find way to compute signature of shortened path via Chen

            # take action
            observation, reward, done, info = env.step(action)
            observation = observation[:,0]
            if printing:
                print("reward:", reward)
            
            episode_inventories.append(info["inventory"])
            episode_reward += reward
            step_counter += 1

            if not done:
                last_tuple_tensor = next_tuple_tensor
            
            if done: # or step_counter % 200 == 0:
                print(
                    "Episode {} | step {} | reward {} | inventory {} ".format(
                        episode, step_counter, episode_reward, episode_inventories[-1]
                    )
                )

        # Record history
        reward_history.append(episode_reward)
        inventory_history.append(episode_inventories)
        action_history.append(episode_actions)

        env.close()

    results = {
        "rewards": reward_history,
        "actions": action_history,
        "inventories": inventory_history,
        "cash": cash_history,
    }
    return results