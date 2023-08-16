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
    
    # compute observations steps if the environment admits an observation interval
    try: 
        do_nothing_steps = (floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("'env' admits no observation interval.")
        do_nothing_steps = 0

    total_step_counter = 0
    
    # TESTING
    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        episode_reward = 0
        episode_inventories = []
        episode_actions = []

        history = deque(maxlen=window_length)

        observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
        history.append(observation)
        
        next_observation_tensor = torch.tensor(
            [observation], requires_grad=False, dtype=torch.float
        )
        last_observation_tensor = next_observation_tensor

        done = False
        do_nothing_counter = 0

        # RUN EPISODE
        while not done:
            if do_nothing_counter < do_nothing_steps:
                # agent only observes 
                action = env.do_nothing_action_id # next action is 'do nothing'
                episode_actions.append(action)
                observation, _, _, _ = env.step(action)
                observation = observation[:,0]
                history.append(observation)

                last_observation_tensor = torch.tensor(
                    [observation], requires_grad=False, dtype=torch.float
                )
                do_nothing_counter += 1
                continue

            if do_nothing_counter == do_nothing_steps:
                # calculate first signature for observed history so far
                history_signature = policy.update_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0)
                )                
            
            # create Q values and select action
            Q = policy(history_signature)
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()

            if len(history) == 1:
                try: action = env.do_nothing_action_id
                except: pass    
            
            episode_actions.append(action)

            # take action
            observation, reward, done, info = env.step(action)
            observation = observation[:,0]
            history.append(observation)
            if printing:
                print("reward:", reward)
            
            next_observation_tensor = torch.tensor(
                [observation], requires_grad=False, dtype=torch.float
            )

            # update signature
            if window_length == None:
                new_path = torch.cat((last_observation_tensor, next_observation_tensor), 0).unsqueeze(0)
                history_signature = policy.update_signature(
                    new_path, last_observation_tensor, history_signature
                )
            else: 
                history_signature = policy.update_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0)
                )

            episode_reward += reward
            episode_inventories.append(observation[1])
            total_step_counter += 1
            if done:
                cash_history.append(info["cash"])                    
            else:
                last_observation_tensor = next_observation_tensor
            
            if done or total_step_counter % 100 == 0:
                print(
                    "\n Episode {} | step {} | reward {} | inventory {}".format(
                        episode, total_step_counter, episode_reward, observation[1]
                    )
                )
                print("Q values:", Q)


        # Record history
        reward_history.append(episode_reward)
        inventory_history.append(episode_inventories)
        action_history.append(episode_actions)

        env.close()

    results = {
        "rewards": reward_history,
        "cash": cash_history,
        "actions": action_history,
        "inventories": inventory_history,
    }
    return results