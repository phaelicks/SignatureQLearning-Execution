import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import tqdm
import copy
from typing import Optional


def train(
        env, 
        policy, 
        episodes, 
        discount: float = 0.99,
        learning_rate: float = 0.1, 
        learning_rate_decay = lambda step: 1,
        epsilon: float = 0.2, 
        epsilon_decay = lambda step: 1,
        window_length: Optional[int] = None
): 
    initial_epsilon: float = epsilon 
    loss_history = []
    reward_history = []
    intermediate_policies = []   

    loss_fn = nn.SmoothL1Loss()
    #loss_fn = nn.MSELoss()

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    
    # compute first_intervall steps to solely observe
    do_nothing_steps = (
        env.first_intervall / env.timestep_duration
    )

    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        episode_loss = 0
        episode_reward = 0

        observation = env.reset() # gym 0.18.0 returns only state at reset
        action = 9 # do nothing
        initial_tuple = np.array([observation, action])

        history = deque()
        history.append(initial_tuple)
        if window_length != None:
            assert window_length > 0, "History window length must be a positive integer."


        initial_tuple_tensor = torch.tensor(
            initial_tuple, requires_grad=False, dtype=torch.float
        ).unsqueeze(0)

        history_signature = policy.update_signature(initial_tuple_tensor)
        last_tuple = initial_tuple

        # run episode
        done = False
        step_counter = 0
        while not done:
            if step_counter < do_nothing_steps:
                action = 9 # do nothing
                observation, _, _, _ = env.step(action)
                history.append(observation)
                if (window_length != None) and (len(history > window_length)):
                    history.popleft()
                step_counter += 1
                continue
            
            
            



                  