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
    decay_step_counter = 0


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
        episode_inventory = []

        observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
        action = 10 # do nothing
        initial_tuple = np.hstack((observation, action / 10))
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
                action = 10 # next action is 'do nothing'
                new_tuple = np.hstack((observation, action / 10))
                history.append(new_tuple)
                episode_actions.append(action)

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
                action = np.random.randint(0, env.action_space.n-1)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            episode_actions.append(action)

            """
            new_t = np.hstack((observation, action))
            new_t_t = torch.tensor([new_t], requires_grad = False, dtype=torch.float)
            new_p = torch.cat((last_tuple_tensor, new_t_t), 0).unsqueeze(0)
            signature_1 = policy.update_signature(new_p, last_tuple_tensor, history_signature)
            Q_selected = policy(signature_1)[0][0]
            """

            # add (o,a) tuple to history
            next_tuple = np.hstack((observation, action / 10))
            next_tuple_tensor = torch.tensor(
                [next_tuple], requires_grad=False, dtype=torch.float
            )                     

            # update history and signature
            history.append(next_tuple) # pops left if maxlen != None               
            if window_length == None:
                new_path = torch.cat((next_tuple_tensor, last_tuple_tensor), 0).unsqueeze(0)
                history_signature = policy.update_signature(new_path, last_tuple_tensor, history_signature)
            else: 
                history_signature = policy.update_signature(
                    torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0)
                )
            #print(history_signature)
            # TODO: find way to compute signature of shortened path via Chen

            # take action
            observation, reward, done, info = env.step(action)
            observation = observation[:,0]
            if printing:
                print("reward: {} | pnl: {} | inv reward: {} | inventory {}".format(
                    reward, info["pnl"], info["inventory_reward"], info["inventory"]
                )
            )

            episode_inventory.append(info["inventory"])

            Q_target = torch.tensor(reward, requires_grad=False, dtype=torch.float)            
            if not done:
                Q1 = policy.create_Q_values(history_signature, next_tuple_tensor, observation)
                maxQ1, _ = torch.max(Q1, -1)
                Q_target += torch.mul(maxQ1, discount)
            Q_target.detach_()
            
            #print(Q[action])
            loss = loss_fn(Q[action], Q_target)
            #loss = loss_fn(Q_selected, Q_target)
            if printing:
                print("loss:", loss)
            policy.zero_grad()
            loss.backward()
            # clip gradient to improve robustness
            #nn.utils.clip_grad_value_(policy.parameters(), 1)
            #nn.utils.clip_grad_norm_(policy.parameters(), 0.25, 2)
            optimizer.step()         
            
            episode_loss += loss.item()
            episode_reward += reward
            step_counter += 1

            # take steps
            scheduler.step()
            decay_step_counter += 1
            epsilon = initial_epsilon * epsilon_decay(decay_step_counter)

            if done:
                cash_history.append(info["cash"])
                terminal_inventory.append(info["inventory"])
            else:
                last_tuple_tensor = next_tuple_tensor
            
            if done or step_counter % 200 == 0:
                print(
                    "Epsiode {} | step {} | reward {} | loss {}".format(
                        episode, step_counter, episode_reward, episode_loss
                    )
                )
                print("Q values:", Q)

        # take steps
        #scheduler.step()
        #epsilon = initial_epsilon * epsilon_decay(episode)

        # Record history
        loss_history.append(episode_loss)
        reward_history.append(episode_reward)
        if (episode+1) % 5 == 0:
            action_history.append(episode_actions)
            inventory_history.append(episode_inventory)
        #action_history.append(episode_actions)
        #inventory_history.append(episode_inventory)

        env.close()

        if (episode+1) % 1000 == 0:
            optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_decay)
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

        if (episode+1) % 10 == 0:
            policy_copy = deepcopy(policy.state_dict())
            intermediate_policies.append(policy_copy)

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






            



                  