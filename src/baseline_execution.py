"""Baseline (naive-sell) policy for execution environments.

Implements a simple deterministic strategy: sell whenever the normalised
inventory exceeds a fixed threshold, and do nothing otherwise.  No learning
model is involved.
"""

import sys
from math import floor
from typing import Optional, Dict, List
import warnings

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
# Main baseline function
# ---------------------------------------------------------------------------

def run_baseline(
    env,
    episodes: int,
    sell_action_id: int = 1,
    debug_mode: Optional[str] = None,
) -> Dict[str, List]:
    """Run a simple baseline sell policy on an execution environment.

    The agent sells one fixed-size order whenever the normalised inventory
    exceeds 50 % of the order size, and does nothing otherwise.

    Args:
        env: A Gym-compatible execution environment.
        episodes: Number of evaluation episodes.
        sell_action_id: Action index corresponding to *sell*.
        debug_mode: 'debug' for step-level, 'info' for episode-level logging.

    Returns:
        Dictionary with evaluation histories::

            {
                "rewards":              [float],
                "cash":                 [float],
                "terminal_inventories": [float],
                "actions":              [[int]],
                "inventories":          [[float]],
                "observations":         [list],
            }
    """
    # --- Result histories (one scalar per episode) ---
    reward_history: List[float] = []
    cash_history: List[float] = []
    terminal_inventory_history: List[float] = []

    # --- Result histories (one sequence per episode) ---
    inventory_histories: List[List[float]] = []
    action_histories: List[List[int]] = []
    observation_histories: List[list] = []

    do_nothing_steps = _compute_do_nothing_steps(env)
    sell_threshold = 0.5 * env.order_fixed_size / env.max_inventory

    total_step_counter = 0

    # ===================================================================
    # Run baseline policy
    # ===================================================================
    pbar = tqdm.trange(episodes, file=sys.stdout)
    for episode in pbar:
        pbar.set_description(f"Episode {episode}")

        episode_reward = 0.0
        episode_actions: List[int] = []
        episode_inventories: List[float] = []
        history: list = []

        observation = env.reset()[:, 0]
        history.append(observation)

        done = False
        do_nothing_counter = 0
        episode_step_counter = 0

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

                do_nothing_counter += 1
                continue

            # Baseline policy: sell if inventory above threshold, else do nothing
            action = (
                sell_action_id
                if observation[-1] > sell_threshold
                else env.do_nothing_action_id
            )
            episode_actions.append(action)

            # Execute action in environment
            observation, reward, done, info = env.step(action)
            observation = observation[:, 0]
            history.append(observation)

            episode_reward += reward
            episode_inventories.append(info["inventory"])

            # Debug logging
            if debug_mode == "debug" and (episode_step_counter % 100 == 0 or done):
                print(
                    f"**** Episode {episode} | Step {episode_step_counter} ****"
                    f"\nSelected action: {action} \nReward: {reward}"
                    f"\nEnvironment info: {info}\n"
                )

            total_step_counter += 1
            episode_step_counter += 1

            if done:
                cash_history.append(info["cash"])
                terminal_inventory_history.append(info["inventory"])

        if debug_mode == "info":
            print(
                f"Episode {episode} | Reward {episode_reward:0.5f} "
                f"| Inventory {terminal_inventory_history[-1]} "
                f"| Steps in run {total_step_counter}"
            )

        # --- Record episode histories ---
        reward_history.append(episode_reward)
        inventory_histories.append(episode_inventories)
        action_histories.append(episode_actions)
        observation_histories.append(history)

        env.close()

    return {
        "rewards": reward_history,
        "cash": cash_history,
        "terminal_inventories": terminal_inventory_history,
        "actions": action_histories,
        "inventories": inventory_histories,
        "observations": observation_histories,
    }
