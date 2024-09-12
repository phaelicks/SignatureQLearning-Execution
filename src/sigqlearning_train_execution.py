from copy import deepcopy
from IPython.display import clear_output
from math import floor
from collections import deque
from time import time
from typing import Optional, Dict, List
import warnings

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
import tqdm.notebook as tqdm

import utils


def train(
        env, 
        policy: nn.Module, 
        episodes: int, 
        discount: float = 0.99,
        learning_rate: float = 0.1, 
        learning_rate_decay = lambda epoch: 1,
        exploration: str = "greedy",
        epsilon: float = 0.2, 
        epsilon_decay = lambda epoch: 1,
        decay_mode: str = "steps",
        window_length: Optional[int] = None,
        debug_mode: Optional[str] = None,
        progress_display: Optional[str] = "progressbar"
) -> Dict[str, List]: 

    # CHECK PROPERTIES
    assert exploration in [
        "greedy",
        "softmax"
    ], "Select 'greedy' or 'softmax' as exploration."

    assert decay_mode in [
        "steps",
        "episodes"
    ], "Select 'steps' or 'episodes' as decay_mode."

    assert (window_length == None) or (
        type(window_length) == int and window_length > 0
    ), "History window length must be a positive integer or None."    

    # PROGESS TRACKING
    if progress_display == 'livelossplot':
        episode_pbar = range(episodes)
        episode_times = [] # workaround to display remaining time in run
        groups = {'Reward': ['reward'], 'Loss': ['loss'], 'Terminal Inventory': ['inventory'], 'Remaining Time (min)': ['time']}
        outputs = [MatplotlibPlot(after_subplot=utils.custom_after_subplot)]
        plotlosses = PlotLosses(groups=groups, outputs=outputs)
    else:
        episode_pbar =  tqdm.trange(episodes)#, file=sys.stdout) # leave=False
        episode_pbar.set_description(f"Episode")

    # LISTS FOR LOGGING HISTORIES
    initial_epsilon: float = epsilon 
    
    # histories over episodes
    loss_history = []
    reward_history = []
    cash_history = []
    terminal_inventory_history = [] 
    first_Q_values_history = []
    last_Q_values_history = []
    
    # per episode histories
    action_histories = []
    inventory_histories = []
    observation_histories = []
    mid_price_histories = []
    
    # histories with different logging interval
    intermediate_policies = []
    
    # GRADIENT DESCENT DETAILS
    loss_fn = nn.SmoothL1Loss()  # alternatively nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_decay)
    
    # compute first_interval steps to solely observe
    try:
        do_nothing_steps = max(1, floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("environment does not admit an observation interval.")
        do_nothing_steps = 0

    # TRAINING
    total_step_counter = 0 # excluding do_nothing_steps
    for episode in episode_pbar:
        start_time = time()

        episode_loss = 0
        episode_reward = 0
        episode_actions = []
        episode_mid_prices = []
        history = deque(maxlen=window_length)

        observation = env.reset()[:,0] # gym 0.18.0 returns state as row vector at reset
        history.append(observation)

        next_observation_tensor = torch.tensor(
            [observation], requires_grad=False, dtype=torch.float
        )
        last_observation_tensor = next_observation_tensor

        done = False
        do_nothing_counter = 0
        episode_step_counter = 0 # excluding do_nothing_steps

        # RUN EPISODE
        while not done:
            # agent only observes 
            if do_nothing_counter < do_nothing_steps:
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
            
            # calculate signature for observed history so far
            if do_nothing_counter == do_nothing_steps:
                history_signature = policy.compute_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0),
                    basepoint=True
                )
                #do_nothing_counter += 1                
            
            # create Q values and select action
            Q = policy(history_signature)[0]
            if exploration == "greedy":
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    _, action = torch.max(Q, -1)
                    action = action.item()
            elif exploration == "softmax":
                probs = F.softmax(Q / epsilon, dim=-1)
                m = Categorical(probs)
                action = m.sample().item()

            # save selected action
            episode_actions.append(action)

            # save first Q value for each episode
            if do_nothing_counter == do_nothing_steps:
                detached_Q = Q.detach()
                first_Q_values_history.append(detached_Q)
                do_nothing_counter += 1  

            # take action
            observation, reward, done, info = env.step(action)
            observation = observation[:,0]
            history.append(observation) # pops left if maxlen != None
            
            episode_reward += reward
            episode_mid_prices.append(info["mid_price"])           
        
            # update signature
            next_observation_tensor = torch.tensor(
                    [observation], requires_grad=False, dtype=torch.float
            )
            if window_length == None:
                new_path = torch.cat((last_observation_tensor, next_observation_tensor), 0).unsqueeze(0)
                history_signature = policy.update_signature(
                    new_path, last_observation_tensor, history_signature
                )
            else: 
                history_signature = policy.compute_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0),
                    basepoint=True
                )
            # TODO: find way to compute signature of shortened path via Chen

            Q_target = torch.tensor(reward, requires_grad=False, dtype=torch.float)            
            if not done:
                Q1 = policy(history_signature)[0]
                maxQ1, _ = torch.max(Q1, -1)
                Q_target += torch.mul(maxQ1, discount)
            Q_target.detach_()
            
            loss = loss_fn(Q[action], Q_target)
            policy.zero_grad()
            loss.backward()
            # TODO: clip gradient here
            optimizer.step()  
            episode_loss += loss.item()

            if debug_mode == "debug":
                if episode_step_counter % 100 == 0 or done:
                    print("""
                        Episode {} | Step {} \n Q values: {} \n Q target: {} \n
                        Selected action: {} \n Loss: {} \n Reward: {} \n Environment info: {} \n
                        """.format(
                            episode, episode_step_counter, Q, Q_target, 
                            action, loss, reward, info
                        )
                    )
            
            episode_step_counter += 1
            total_step_counter += 1

            if decay_mode == "steps":
                # take steps
                scheduler.step()
                epsilon = initial_epsilon * epsilon_decay(total_step_counter)

            if done:
                cash_history.append(info["cash"])
                terminal_inventory_history.append(info["inventory"])
                last_Q_values_history.append(Q.detach())
                if decay_mode == "episodes":
                    # take steps
                    scheduler.step()
                    epsilon = initial_epsilon * epsilon_decay(episode+1)
            else:
                last_observation_tensor = next_observation_tensor

        env.close()

        # Record histories
        loss_history.append(episode_loss)
        reward_history.append(episode_reward)
        action_histories.append(episode_actions)
        observation_histories.append(history)
        mid_price_histories.append(episode_mid_prices)

        # record intermediate Q function each 10 episodes and at end
        if (episode+1) % 10 == 0 or (episode+1) == episodes:
            #policy_copy = deepcopy(policy.state_dict())
            intermediate_policies.append(deepcopy(policy.state_dict()))

        # plot intermediate results to see progress
        if (episode+1) % 50 == 0 or (episode+1) == episodes:
            #clear_output(wait=True)
            # TODO: find way to clear plots but keep progress bar in notebook
            utils.plot_results([
                reward_history,
                loss_history,
                cash_history,
                terminal_inventory_history,
            ], size=(7,5))

        # print episode statistics if needed
        if debug_mode == "info":
            print("Episode {} | Reward {} | Loss {} | Steps in run {}".format(
                    episode, episode_reward, episode_loss, total_step_counter
                )
            )

        # progress display with livelossplot
        if progress_display == 'livelossplot':
            episode_times.append(abs(start_time - time()))
            mean_time = np.mean(episode_times)
            plotlosses.update({
                'reward': episode_reward,
                'loss': episode_loss,
                'inventory': terminal_inventory_history[-1],
                'time': mean_time * (episodes - episode) / 60
            })
            plotlosses.send()
        
    results = {
        "rewards": reward_history,
        "losses": loss_history,
        "cash": cash_history,
        "terminal_inventory": terminal_inventory_history,
        "actions": action_histories,
        "inventories": inventory_histories,
        "observations": observation_histories,
        "intermediate": intermediate_policies,
        "mid_prices": mid_price_histories,
        "first_Q_values": first_Q_values_history,
    }
    return results






            



                  