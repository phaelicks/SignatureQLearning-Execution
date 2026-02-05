import numpy as np
import torch
from collections import deque
import sys
import tqdm.notebook as tqdm
from typing import Optional, Dict, List
from math import floor
import warnings


def test(
        env, 
        qfunction, 
        episodes, 
        epsilon: float = 0.0, 
        window_length: Optional[int] = None,
        debug_mode: Optional[str] = None,
) -> Dict[str, List]: 
    # history over episodes
    reward_history = []
    cash_history = []
    terminal_inventory_history = []
    terminal_wealth_history = []
    first_obs_value_history = []

    # per episode histories
    inventory_histories = []
    action_histories = []
    observation_histories = []
    mid_price_histories = []

    # CHECK PROPERTIES
    assert (window_length == None) or (
        type(window_length) == int and window_length > 0
    ), "History window length must be a positive integer or None."
    
    # compute observations steps if the environment admits an observation interval
    try: 
        do_nothing_steps = max(1, floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("environment admits no observation interval.")
        do_nothing_steps = 0

    total_step_counter = 0
    
    # TESTING
    qfunction.eval()
    with torch.no_grad():

        pbar = tqdm.trange(episodes, file=sys.stdout)
        for episode in pbar:
            pbar.set_description(f"Episode {episode}")

            episode_reward = 0
            episode_actions = []
            episode_mid_prices = []
            episode_inventories = []

            history = deque(maxlen=window_length)

            observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
            history.append(observation)
            
            next_observation_tensor = torch.tensor(
                [observation], requires_grad=False, dtype=torch.float
            )
            last_observation_tensor = next_observation_tensor

            done = False
            do_nothing_counter = 0
            episode_step_counter = 0

            # RUN EPISODE
            while not done:
                if do_nothing_counter < do_nothing_steps:
                    # agent only observes 
                    action = env.do_nothing_action_id # next action is 'do nothing'
                    episode_actions.append(action)

                    observation, _, _, info = env.step(action)
                    observation = observation[:,0]
                    history.append(observation)
                    episode_inventories.append(info["inventory"])
                    episode_mid_prices.append(info["mid_price"])

                    last_observation_tensor = torch.tensor(
                        [observation], requires_grad=False, dtype=torch.float
                    )
                    do_nothing_counter += 1
                    continue

                if do_nothing_counter == do_nothing_steps:
                    # calculate signature and value function of observed history so far
                    # for observation = (time, inventory), this history is always the same
                    history_signature = qfunction.compute_signature(
                        torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0),
                    )
                    first_obs_value = max(qfunction(history_signature)[0].detach()).item()
                    do_nothing_counter += 1         
                
                # create Q values and select action
                Q = qfunction(history_signature)[0]
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    _, action = torch.max(Q, -1)
                    action = action.item()
                
                episode_actions.append(action)

                # take action
                observation, reward, done, info = env.step(action)
                observation = observation[:,0]
                history.append(observation)
                
                episode_reward += reward
                episode_mid_prices.append(info["mid_price"])
                episode_inventories.append(info["inventory"])

                if debug_mode == "debug":
                    if episode_step_counter % 100 == 0 or done:
                        print(
                            "\
                            Episode {} | Step {} \n Q values: {} \n \
                            Selected action: {} \n Reward: {} \n Environment info: {} \n \
                            ".format(
                                episode, episode_step_counter, Q, 
                                action, reward, info
                            )
                        )
                
                # update signature
                next_observation_tensor = torch.tensor(
                    [observation], requires_grad=False, dtype=torch.float
                )
                if window_length == None:
                    new_path = torch.cat((last_observation_tensor, next_observation_tensor), 0).unsqueeze(0)
                    history_signature = qfunction.update_signature(
                        new_path, last_observation_tensor, history_signature
                    )
                else: 
                    history_signature = qfunction.compute_signature(
                        torch.tensor(history, requires_grad=False, dtype=torch.float).unsqueeze(0),
                    )

                total_step_counter += 1
                episode_step_counter += 1

                if done:
                    cash_history.append(info["cash"])  
                    terminal_inventory_history.append(info["inventory"])                                  
                    try:
                        terminal_wealth_history.append(info["marked_to_market"])
                    except:
                        pass
                else:
                    last_observation_tensor = next_observation_tensor

            if debug_mode == "info":
                print("Episode {0} | Reward {1:0.5f} | Inventory {2} | Steps in run {3}".format(
                    episode, episode_reward, terminal_inventory_history[-1], total_step_counter
                )
            )            

            # Record history
            reward_history.append(episode_reward)
            inventory_histories.append(episode_inventories)
            first_obs_value_history.append(first_obs_value)
            action_histories.append(episode_actions)
            observation_histories.append(history)
            mid_price_histories.append(episode_mid_prices)

            env.close()

    results = {
        "rewards": reward_history,
        "cash": cash_history,
        "terminal_inventories": terminal_inventory_history,
        "first_obs_values": first_obs_value_history,
        "terminal_wealth": terminal_wealth_history,
        "actions": action_histories,
        "inventories": inventory_histories,
        "observations": observation_histories,
        "mid_prices": mid_price_histories,
    }
    return results