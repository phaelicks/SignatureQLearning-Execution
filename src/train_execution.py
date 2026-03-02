"""Training loop for Q-learning in execution environments.

Works with any Q-function model that implements the unified interface:
- 'forward(history)' → Q-values tensor of shape '(1, n_actions)'
- 'reset_state()'    → clears internal model state, may be void

Supports epsilon-greedy and softmax exploration, configurable decay schedules,
and multiple progress display modes (tqdm progress bar, livelossplot).
"""

from copy import deepcopy
from collections import deque
from math import floor
from time import time
from typing import Optional, Dict, List, Any
import warnings

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F
import tqdm.notebook as tqdm

import utils

# ---------------------------------------------------------------------------
# Valid configuration options
# ---------------------------------------------------------------------------
EXPLORATION_MODES = ("greedy", "softmax")
DECAY_MODES = ("steps", "episodes")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _select_action(Q, exploration, epsilon, n_actions):
    """Select an action from Q-values using the specified exploration strategy.

    Args:
        Q: Q-value tensor of shape '(n_actions,)'.
        exploration: 'greedy' for epsilon-greedy, 'softmax' for Boltzmann.
        epsilon: Exploration parameter (probability for greedy, temperature for softmax).
        n_actions: Number of available actions.

    Returns:
        Selected action index (int).
    """
    if exploration == "greedy":
        if np.random.rand() < epsilon:
            return np.random.randint(0, n_actions)
        _, action = torch.max(Q, -1)
        return action.item()
    elif exploration == "softmax":
        probs = F.softmax(Q / epsilon, dim=-1)
        return Categorical(probs).sample().item()


def _validate_parameters(exploration, decay_mode, window_length):
    """Validate training hyperparameters."""
    assert exploration in EXPLORATION_MODES, (
        f"Select one of {EXPLORATION_MODES} as exploration."
    )
    assert decay_mode in DECAY_MODES, (
        f"Select one of {DECAY_MODES} as decay_mode."
    )
    assert (window_length is None) or (
        isinstance(window_length, int) and window_length > 0
    ), "History window length must be a positive integer or None."


def _compute_do_nothing_steps(env):
    """Compute the number of initial observation-only steps from the environment."""
    try:
        return max(1, floor(env.observation_interval / env.timestep_duration))
    except AttributeError:
        warnings.warn("Environment does not admit an observation interval.")
        return 0


def _setup_progress_display(progress_display, episodes):
    """Initialize the progress display mechanism.

    Returns:
        Tuple of '(episode_pbar, plotlosses_or_None)'.
    """
    if progress_display == "livelossplot":
        episode_pbar = range(episodes)
        groups = {
            "Reward": ["reward"],
            "Loss": ["loss"],
            "Terminal Inventory": ["inventory"],
            "Remaining Minutes in Run": ["time"],
        }
        outputs = [MatplotlibPlot(after_subplot=utils.custom_after_subplot)]
        plotlosses = PlotLosses(groups=groups, outputs=outputs)
        return episode_pbar, plotlosses
    else:
        episode_pbar = tqdm.trange(episodes, leave=False)
        episode_pbar.set_description("Episode")
        return episode_pbar, None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    env,
    qfunction: nn.Module,
    episodes: int,
    discount: float = 0.99,
    learning_rate: float = 0.1,
    learning_rate_decay: Optional[Dict[str, Any]] = None,
    exploration: str = "greedy",
    epsilon: float = 0.8,
    epsilon_decay: Optional[Dict[str, Any]] = None,
    decay_mode: str = "steps",
    window_length: Optional[int] = None,
    debug_mode: Optional[str] = None,
    progress_display: Optional[str] = "progressbar",
) -> Dict[str, List]:
    """Train a Q-function on an execution environment.

    This procedure is **model-agnostic**: any Q-function model that accepts an
    observation history and returns Q-values can be plugged in.  Model-specific
    feature extraction (e.g., path-signature computation for 'SigQFunction')
    is handled internally by the model's 'forward' method.

    Args:
        env: A Gym-compatible execution environment
        qfunction: A Q-function module implementing 'forward(history)' and 'reset_state()'
        episodes: Number of training episodes
        discount: Discount factor for future rewards
        learning_rate: Initial learning rate for the Adam optimiser
        learning_rate_decay: Config dict for learning-rate schedule. Defaults to no decay
        exploration: Exploration strategy 'greedy' (ε-greedy) or 'softmax'.
        epsilon: Exploration strategy starting parameter (probability or temperature)
        epsilon_decay: Config dict for epsilon decay schedule. Defaults to no decay.
        decay_mode: Whether to apply decay per 'steps' or per 'episodes'.
        window_length: Sliding-window size for the observation history.
            Defaults to 'None' which keeps the full history.
        debug_mode: 'debug' for step-level, 'info' for episode-level logging.
        progress_display: 'progressbar' for tqdm, 'livelossplot' for live plots.

    Returns:
        Dictionary with training histories::

            {
                "rewards":              [float],
                "losses":               [float],
                "cash":                 [float],
                "terminal_inventories": [float],
                "actions":              [[int]],
                "observations":         [[float]],
                "mid_prices":           [[float]],
                "first_obs_values":     [float],
                "intermediate":         [state_dict],
            }
    """
    # --- Default mutable arguments ---
    if learning_rate_decay is None:
        learning_rate_decay = dict(mode=None)
    if epsilon_decay is None:
        epsilon_decay = dict(mode=None)

    _validate_parameters(exploration, decay_mode, window_length)

    # --- Progress tracking ---
    episode_pbar, plotlosses = _setup_progress_display(progress_display, episodes)
    episode_times: List[float] = [] # workaround to display estimated remaining time in live chart

    # --- Result histories (one scalar per episode) ---
    loss_history: List[float] = []
    reward_history: List[float] = []
    cash_history: List[float] = []
    terminal_inventory_history: List[float] = []
    first_obs_value_history: List[float] = []
    last_Q_values_history: List[torch.Tensor] = []

    # --- Result histories (one sequence per episode) ---
    action_histories: List[List[int]] = []
    observation_histories: List[list[float]] = []
    mid_price_histories: List[List[float]] = []

    # --- Periodic snapshots ---
    intermediate_qfunctions: List[dict] = []

    # --- Exploration schedule ---
    initial_epsilon: float = epsilon
    epsilon_decay_fn = utils.create_decay_schedule(
        start_value=initial_epsilon, **epsilon_decay
    )

    # --- Optimiser and scheduler ---
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(qfunction.parameters(), lr=learning_rate)
    lr_decay_fn = utils.create_decay_schedule(
        start_value=learning_rate, **learning_rate_decay
    )
    scheduler_fn = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_fn)

    do_nothing_steps = _compute_do_nothing_steps(env)

    # ===================================================================
    # Training loop
    # ===================================================================
    total_step_counter = 0  # excluding do-nothing steps

    for episode in episode_pbar:
        start_time = time() # workaround to display estimated remaining time in live chart

        episode_loss = 0.0
        episode_reward = 0.0
        episode_actions: List[int] = []
        episode_mid_prices: List[float] = []
        history: deque = deque(maxlen=window_length)

        observation = env.reset()[:, 0]
        history.append(observation)

        qfunction.reset_state() # necessary to clear internal state in 'SigQFunction'

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

                observation, _, _, _ = env.step(action)
                observation = observation[:, 0]
                history.append(observation)

                do_nothing_counter += 1
                continue

            # Compute Q-values and select action
            Q = qfunction(history)[0]

            if episode_step_counter == 0:
                first_obs_value = max(Q.detach()).item()

            action = _select_action(Q, exploration, epsilon, env.action_space.n)
            episode_actions.append(action)

            # Execute action in environment
            observation, reward, done, info = env.step(action)
            observation = observation[:, 0]
            history.append(observation)
            episode_reward += reward
            episode_mid_prices.append(info["mid_price"])

            # Compute TD target
            Q_target = torch.tensor(reward, requires_grad=False, dtype=torch.float)
            if not done:
                Q_next = qfunction(history)[0]
                max_Q_next, _ = torch.max(Q_next, -1)
                Q_target += discount * max_Q_next
            Q_target.detach_()

            # Gradient step
            loss = loss_fn(Q[action], Q_target)
            episode_loss += loss.item()
            loss.backward()
            optimizer.step()
            qfunction.zero_grad()

            # Debug logging
            if debug_mode == "debug" and (episode_step_counter % 60 == 0 or done):
                print(
                    f"\n**** Episode {episode} | Step {episode_step_counter+1} ****"
                    f"\nQ values: {Q} \nQ target: {Q_target}"
                    f"\nSelected action: {action} \nLoss: {loss}"
                    f"\nReward: {reward} \nEnvironment info: {info}\n"
                )
            
            episode_step_counter += 1
            total_step_counter += 1

            # Decay schedules (per step)
            if decay_mode == "steps":
                scheduler_fn.step()
                epsilon = initial_epsilon * epsilon_decay_fn(total_step_counter)

            if done:
                cash_history.append(info["cash"])
                terminal_inventory_history.append(info["inventory"])
                last_Q_values_history.append(Q.detach())
                if decay_mode == "episodes":
                    scheduler_fn.step()
                    epsilon = initial_epsilon * epsilon_decay_fn(episode + 1)

        env.close()

        # --- Record episode histories ---
        loss_history.append(episode_loss)
        reward_history.append(episode_reward)
        first_obs_value_history.append(first_obs_value)
        action_histories.append(episode_actions)
        observation_histories.append(list(history))
        mid_price_histories.append(episode_mid_prices)

        # Snapshot Q-function periodically each 10 episodes and at end of training
        if (episode + 1) % 10 == 0 or (episode + 1) == episodes:
            intermediate_qfunctions.append(deepcopy(qfunction.state_dict()))

        if debug_mode == "info":
            print(
                f"Episode {episode} | Reward {episode_reward} "
                f"| Loss {episode_loss} | Steps in run {total_step_counter}"
            )

        # Live loss-plot update
        if progress_display == "livelossplot" and plotlosses is not None:
            episode_times.append(abs(start_time - time()))
            mean_time = np.mean(episode_times)
            plotlosses.update({
                "reward": episode_reward,
                "loss": episode_loss,
                "inventory": terminal_inventory_history[-1],
                "time": mean_time * (episodes - episode) / 60,
            })
            plotlosses.send()

    return {
        "rewards": reward_history,
        "losses": loss_history,
        "cash": cash_history,
        "terminal_inventories": terminal_inventory_history,
        "actions": action_histories,
        "observations": observation_histories,
        "mid_prices": mid_price_histories,
        "first_obs_values": first_obs_value_history,
        "intermediate": intermediate_qfunctions,
    }
