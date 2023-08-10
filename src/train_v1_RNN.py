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


def train(
        env, 
        policy, 
        episodes, 
        discount: float = 0.99,
        learning_rate: float = 0.1, 
        learning_rate_decay = lambda step: 1,
        epsilon: float = 0.2, 
        epsilon_decay = lambda step: 1,
        window_length: Optional[int] = None,
        printing: bool = False
) -> Dict[str, List]: 
    initial_epsilon: float = epsilon 
    loss_history = []
    reward_history = []
    cash_history = []
    terminal_inventory = []
    intermediate_policies = []   
    action_history = []
    inventory_history = []
    midprice_history = []

    # CHECK PROPERTIES
    if window_length != None:
        assert window_length > 0, "History window length must be a positive integer."
    
    # GRADIENT DESCENT DETAILS
    loss_fn = nn.SmoothL1Loss() # nn.MSELoss()

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    
    # compute first_interval steps to solely observe
    do_nothing_steps = (
        floor(env.observe_interval / env.timestep_duration)
    )

    # TRAINING
    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        episode_loss = 0
        episode_reward = 0
        episode_actions = []
        episode_inventories = []
        episode_midprices = []

        observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
        action = 2 # do nothing
        episode_actions.append(action)

        history = deque(maxlen=window_length)

        done = False
        step_counter = 0
        do_nothing_counter = 0

        # RUN EPISODE
        while not done:
            if do_nothing_counter < do_nothing_steps:
                history.append(observation)
                # agent only observes 
                action = 2 # next action is 'do nothing'
                episode_actions.append(action)

                observation, _, _, _ = env.step(action)
                observation = observation[:,0]

                do_nothing_counter += 1
                continue

            history.append(observation)
            history_tensor = torch.tensor(
                history, requires_grad=False, dtype=torch.float
            ).unsqueeze(0)
            
            # create Q values and select action
            Q = policy(history_tensor)[0]
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            episode_actions.append(action)

            # take action
            observation_1, reward, done, info = env.step(action)
            observation_1 = observation_1[:,0]
            if printing:
                print("reward: {} | pnl: {} | inventory {}".format(
                    reward, info["pnl"], info["inventory"]
                )
                )
                if done: 
                    print("update reward {}".format(reward - info["pnl"]))

            episode_inventories.append(info["inventory"])                

            # update history 
            history_1 = history
            history_1.append(observation_1)
            history_1_tensor = torch.tensor(
                history_1, requires_grad=False, dtype=torch.float
            ).unsqueeze(0)
            
            Q_target = torch.tensor(reward, requires_grad=False, dtype=torch.float)            
            if not done:
                Q1 = policy(history_1_tensor)[0]
                maxQ1, _ = torch.max(Q1, -1)
                Q_target += torch.mul(maxQ1, discount)
            Q_target.detach_()
            
            #print(Q[action])
            loss = loss_fn(Q[action], Q_target)
            if printing:
                print("loss:", loss)
            policy.zero_grad()
            loss.backward()
            # clip gradient to improve robustness
            #nn.utils.clip_grad_value_(policy.parameters(), 0.5)
            #nn.utils.clip_grad_norm_(policy.parameters(), 0.1, 2)
            optimizer.step()         
            
            episode_loss += loss.item()
            episode_reward += reward
            step_counter += 1

            # take steps
            scheduler.step()
            epsilon = initial_epsilon * epsilon_decay(step_counter)

            if done:
                cash_history.append(info["cash"])
                terminal_inventory.append(info["inventory"])
            else:
                observation = observation_1
            
            if done or step_counter % 100 == 0:
                print(
                    "\n Episode {} | step {} | reward {} | loss {}".format(
                        episode, step_counter, episode_reward, episode_loss
                    )
                )
                print("Q values:", Q)

        env.close()

        # Record history
        loss_history.append(episode_loss)
        reward_history.append(episode_reward)
        #if (episode+1) % 5 == 0 or episode == 0:
        #    action_history.append(episode_actions)
        #    inventory_history.append(episode_inventories)
        #    midprice_history.append(episode_midprices)
        action_history.append(episode_actions)
        inventory_history.append(episode_inventories)
        midprice_history.append(episode_midprices)

        if (episode+1) % 10 == 0:
            policy_copy = deepcopy(policy.state_dict())
            intermediate_policies.append(policy_copy)

        if (episode+1) % 1000 == 0:
            optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_decay)
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

    results = {
        "rewards": reward_history,
        "losses": loss_history,
        "cash": cash_history,
        "terminal_inventory": terminal_inventory,
        "actions": action_history,
        "inventories": inventory_history,
        "history": history,
        "intermediate": intermediate_policies,
    }
    return results






            



                  