"""Evaluation loop for Q-learning in execution environments.

Works with any Q-function model that implements the unified interface:
- 'forward(history)' → Q-values tensor of shape '(1, n_actions)'
- 'reset_state()'    → clears internal model state
"""

import sys
from collections import deque
from math import floor
from typing import Optional, Dict, List
import warnings

import numpy as np
import torch
import tqdm.notebook as tqdm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_do_nothing_steps(env):
    """Compute the number of initial observation-only steps from the environment."""
    try:
        return max(1, floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("Environment does not admit an observation interval.")
        return 0


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test(
    env,
    qfunction,
    episodes: int,
    epsilon: float = 0.0,
    window_length: Optional[int] = None,
    debug_mode: Optional[str] = None,
) -> Dict[str, List]:
    """Evaluate a trained Q-function on an execution environment.

    This procedure is **model-agnostic**: any Q-function model that accepts an
    observation history and returns Q-values can be plugged in.

    Args:
        env: A Gym-compatible execution environment.
        qfunction: A trained Q-function module implementing 'forward(history)' and 'reset_state()'.
        episodes: Number of evaluation episodes.
        epsilon: Probability of random action in epsilon-greedy. Defaults to '0.0' for purely greedy.
        window_length: Sliding-window size for the observation history.
            Defaults to 'None' which keeps the full history.
        debug_mode: 'debug' for step-level, 'info' for episode-level logging.

    Returns:
        Dictionary with evaluation histories::

            {
                "rewards":              [float],
                "cash":                 [float],
                "terminal_inventories": [float],
                "first_obs_values":     [float],
                "terminal_wealth":      [float],
                "actions":              [[int]],
                "inventories":          [[float]],
                "observations":         [[float]],
                "mid_prices":           [[float]],
            }
    """
    assert (window_length is None) or (
        isinstance(window_length, int) and window_length > 0
    ), "History window length must be a positive integer or None."

    do_nothing_steps = _compute_do_nothing_steps(env)

    # --- Result histories (one scalar per episode) ---
    reward_history: List[float] = []
    cash_history: List[float] = []
    terminal_inventory_history: List[float] = []
    terminal_wealth_history: List[float] = []
    first_obs_value_history: List[float] = []

    # --- Result histories (one sequence per episode) ---
    inventory_histories: List[List[float]] = []
    action_histories: List[List[int]] = []
    observation_histories: List[List[float]] = []
    mid_price_histories: List[List[float]] = []

    total_step_counter = 0

    # ===================================================================
    # Evaluation loop
    # ===================================================================
    qfunction.eval()
    with torch.no_grad():

        pbar = tqdm.trange(episodes, file=sys.stdout)
        for episode in pbar:
            pbar.set_description(f"Episode {episode}")

            episode_reward = 0.0
            episode_actions: List[int] = []
            episode_mid_prices: List[float] = []
            episode_inventories: List[float] = []

            history: deque = deque(maxlen=window_length)

            observation = env.reset()[:, 0]
            history.append(observation)

            qfunction.reset_state()

            done = False
            do_nothing_counter = 0
            episode_step_counter = 0
            first_obs_value = 0.0

            # --- Run episode ---
            while not done:
                # Observation-only phase
                if do_nothing_counter < do_nothing_steps:
                    action = env.do_nothing_action_id
                    episode_actions.append(action)

                    observation, _, _, info = env.step(action)
                    observation = observation[:, 0]
                    history.append(observation)
                    episode_inventories.append(info["inventory"])
                    episode_mid_prices.append(info["mid_price"])

                    do_nothing_counter += 1
                    continue

                # Compute Q-values and select action (epsilon-greedy)
                Q = qfunction(history)[0]

                if episode_step_counter == 0:
                    first_obs_value = max(Q).item()

                if np.random.rand() < epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    _, action = torch.max(Q, -1)
                    action = action.item()

                episode_actions.append(action)

                # Execute action in environment
                observation, reward, done, info = env.step(action)
                observation = observation[:, 0]
                history.append(observation)

                episode_reward += reward
                episode_mid_prices.append(info["mid_price"])
                episode_inventories.append(info["inventory"])

                # Debug logging
                if debug_mode == "debug" and (episode_step_counter % 60 == 0 or done):
                    print(
                        f"**** Episode {episode} | Step {episode_step_counter} ****"
                        f"\nQ values: {Q} \nSelected action: {action}"
                        f"\nReward: {reward} \nEnvironment info: {info}\n"
                    )

                total_step_counter += 1
                episode_step_counter += 1

                if done:
                    cash_history.append(info["cash"])
                    terminal_inventory_history.append(info["inventory"])
                    if "marked_to_market" in info:
                        terminal_wealth_history.append(info["marked_to_market"])

            if debug_mode == "info":
                print(
                    f"Episode {episode} | Reward {episode_reward:0.5f} "
                    f"| Inventory {terminal_inventory_history[-1]} "
                    f"| Steps in run {total_step_counter}"
                )

            # --- Record episode histories ---
            reward_history.append(episode_reward)
            inventory_histories.append(episode_inventories)
            first_obs_value_history.append(first_obs_value)
            action_histories.append(episode_actions)
            observation_histories.append(list(history))
            mid_price_histories.append(episode_mid_prices)

            env.close()

    return {
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
