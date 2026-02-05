import sys
import tqdm.notebook as tqdm
from typing import Optional, Dict, List
from math import floor
import warnings


def run_baseline(
        env, 
        episodes,
        sell_action_id: int = 1, 
        debug_mode: Optional[str] = None,
) -> Dict[str, List]: 
    # history over episodes
    reward_history = []
    cash_history = []
    terminal_inventory_history = []

    # per episode histories
    inventory_histories = []
    action_histories = []
    observation_histories = []

    # CHECK PROPERTIES

    # compute observations steps if the environment admits an observation interval
    try: 
        do_nothing_steps = max(1, floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("environment admits no observation interval.")
        do_nothing_steps = 0

    total_step_counter = 0
    
    # RUN BASELINE POLICY

    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        pbar.set_description(f"Episode {episode}")

        episode_reward = 0
        episode_actions = []
        episode_inventories = []

        history = []

        observation = env.reset()[:,0] # as row vector, gym 0.18.0 returns only state at reset
        history.append(observation)
        
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

                do_nothing_counter += 1
                continue 

            # sell as long as absolute inventory above 50% of fixed order size
            threshold = 0.5 * env.order_fixed_size/env.max_inventory
            action = sell_action_id if observation[-1] > threshold else env.do_nothing_action_id
            episode_actions.append(action)

            # take action
            observation, reward, done, info = env.step(action)
            observation = observation[:,0]
            history.append(observation)
            
            episode_reward += reward
            episode_inventories.append(info["inventory"])

            if debug_mode == "debug":
                if episode_step_counter % 100 == 0 or done:
                    print(
                        "\
                        Episode {} | Step {} \n Selected action: {} \n \
                        Reward: {} \n Environment info: {} \n \
                        ".format(
                            episode, episode_step_counter,
                            action, reward, info
                        )
                    )
            
            total_step_counter += 1
            episode_step_counter += 1

            if done:
                cash_history.append(info["cash"])  
                terminal_inventory_history.append(info["inventory"])                                  


        if debug_mode == "info":
            print("Episode {0} | Reward {1:0.5f} | Inventory {2} | Steps in run {3}".format(
                episode, episode_reward, terminal_inventory_history[-1], total_step_counter
            )
        )            

        # Record history
        reward_history.append(episode_reward)
        inventory_histories.append(episode_inventories)
        action_histories.append(episode_actions)
        observation_histories.append(history)

        env.close()

    results = {
        "rewards": reward_history,
        "cash": cash_history,
        "terminal_inventories": terminal_inventory_history,
        "actions": action_histories,
        "inventories": inventory_histories,
        "observations": observation_histories,
    }
    return results