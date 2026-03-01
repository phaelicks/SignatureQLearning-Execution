"""Q-function architectures for reinforcement learning in execution environments.

All Q-function models implement a unified interface:
- forward(history) accepts the observation history (deque or list of numpy arrays)
  and returns Q-values as a tensor of shape (1, n_actions).
- reset_state() clears any internal state; call at the start of each episode.

This design allows the training and testing procedures to work with any model
without knowledge of model-specific internals (e.g., path signature computation).
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import signatory

class AbstractQFunction(nn.Module, ABC):
    """Abstract base class for Q-function models."""    
    
    def reset_state(self):
        """Reset any internal state. 
        
        Default: no-op. Override if the model maintains internal state (e.g., cached signatures) 
        """
        pass

    @abstractmethod
    def forward(self, history):
        """Compute Q-values from an observation history.

        Args:
            history: Observation sequence (deque or list), each entry of shape '(in_channels,)'.

        Returns:
            Q-values tensor of shape '(1, out_dimension)'.
        """
        pass


class SigQFunction(AbstractQFunction):
    """Q-function based on the path signature of the observation history.

    The observation history is internally converted to its truncated path signature,
    which is then mapped to Q-values via a linear layer. Signature computations are
    cached and incrementally updated via Chen's identity when possible, avoiding
    redundant recomputations across consecutive forward calls.

    The model maintains internal state (the cached signature). Call 'reset_state()'
    at the start of each episode to clear this cache.

    Args:
        env: Gym-compatible environment with 'observation_space' and 'action_space'.
        sig_depth: Truncation depth of the path signature.
        in_channels: Dimension of each observation vector. Defaults to env obs dim.
        out_dimension: Number of output Q-values. Defaults to env action count.
        basepoint: Basepoint for signature computation, either a numeric value / tensor
            (used as fixed basepoint), 'True' (zero basepoint), or 'False'/'None' (no basepoint).
        initial_bias: Initial value for the linear layer bias. 'None' to skip.
    """

    def __init__(self, env, sig_depth, in_channels=None, out_dimension=None,
                 basepoint=True, initial_bias=0.01):
        assert (
            env.observation_space.shape[1] == 1
        ), "Observation space variables must be scalars."

        super().__init__()

        self.sig_depth = sig_depth
        self.in_channels = (
            env.observation_space.shape[0] if in_channels is None else in_channels
        )
        self.out_dimension = (
            env.action_space.n if out_dimension is None else out_dimension
        )
        self.basepoint = (
            torch.tensor(basepoint, requires_grad=False, dtype=torch.float).unsqueeze(0)
            if basepoint not in (None, False, True) else basepoint
        )
        self.initial_bias = initial_bias

        self.sig_channels = signatory.signature_channels(
            channels=self.in_channels, depth=sig_depth
        )
        self.linear = nn.Linear(self.sig_channels, self.out_dimension, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.initial_bias is not None:
            self.linear.bias.data.fill_(self.initial_bias)

        # Internal cache for incremental signature updates
        self._cached_signature = None
        self._cached_history_len = 0
        self._last_observation_tensor = None

    def reset_state(self):
        """Reset cached signature state. Call at the start of each episode."""
        self._cached_signature = None
        self._cached_history_len = 0
        self._last_observation_tensor = None

    def forward(self, history):
        """Compute Q-values from an observation history.

        Internally converts the history to its truncated path signature and
        maps it to Q-values through a linear layer. Uses Chen's identity for
        efficient incremental updates when the history grew by exactly one
        observation since the last call.

        Args:
            history: Observation sequence (deque or list), each entry of
                shape '(in_channels,)'.

        Returns:
            Q-values tensor of shape '(1, out_dimension)'.
        """
        history_list = list(history)
        current_len = len(history_list)
        current_last_obs = torch.tensor(
            [history_list[-1]], requires_grad=False, dtype=torch.float
        )

        signature = self._resolve_signature(history_list, current_len, current_last_obs)

        # Update cache
        self._cached_signature = signature.detach()
        self._cached_history_len = current_len
        self._last_observation_tensor = current_last_obs

        return self.linear(signature)

    # ------------------------------------------------------------------
    # Internal signature helpers
    # ------------------------------------------------------------------

    def _resolve_signature(self, history_list, current_len, current_last_obs):
        """Compute or retrieve the path signature, using caching where possible.

        Caching strategy (checked in order):
        1. *Reuse* — history unchanged since last call (same length and last obs).
        2. *Incremental update* — history grew by exactly one observation;
           extend the cached signature via Chen's identity.
        3. *Full recomputation* — all other cases (first call, window shift, etc.).
        """
        # No cache available — full computation
        if self._cached_signature is None:
            return self._compute_signature(
                torch.tensor(
                    history_list, requires_grad=False, dtype=torch.float
                ).unsqueeze(0)
            )

        # History unchanged since last call — reuse cached signature
        if (current_len == self._cached_history_len
                and self._last_observation_tensor is not None
                and torch.equal(current_last_obs, self._last_observation_tensor)):
            return self._cached_signature

        # History grew by one observation — incremental update (Chen's identity)
        if (current_len == self._cached_history_len + 1
                and self._last_observation_tensor is not None):
            new_path = torch.cat(
                (self._last_observation_tensor, current_last_obs), dim=0
            ).unsqueeze(0)
            return self._incremental_update(
                new_path, self._last_observation_tensor, self._cached_signature
            )

        # Fallback — full recomputation (e.g., window shifted)
        return self._compute_signature(
            torch.tensor(
                history_list, requires_grad=False, dtype=torch.float
            ).unsqueeze(0)
        )

    def _compute_signature(self, path):
        """Compute the truncated path signature.

        Args:
            path: Tensor of shape '(batch, length, in_channels)'.

        Returns:
            Signature tensor of shape '(batch, sig_channels)'.
        """
        if path.shape[1] == 1 and self.basepoint in (None, False):
            return signatory.signature(
                path=path, depth=self.sig_depth, basepoint=path.squeeze(0)
            )
        return signatory.signature(
            path=path, depth=self.sig_depth, basepoint=self.basepoint
        )

    def _incremental_update(self, new_path, last_basepoint, signature):
        """Extend a signature incrementally using Chen's identity.

        Given 'S(X)' and a new segment 'Y' with 'Y[0] = X[-1]',
        returns the signature of the concatenated pth 'S(X * Y)'.

        Args:
            new_path: Path segment '[last_obs, new_obs]'. Shape '(batch, 2, in_channels)'.
            last_basepoint: Last point of previous path. Shape '(1, in_channels)'.
            signature: Previous signature. Shape '(batch, sig_channels)'.

        Returns:
            Updated signature tensor of shape '(batch, sig_channels)'.
        """
        return signatory.signature(
            path=new_path, depth=self.sig_depth,
            basepoint=last_basepoint, initial=signature,
        )


class RNNQFunction(AbstractQFunction):
    """Q-function using a vanilla RNN to process the observation sequence.

    Args:
        env: Gym-compatible environment.
        layers: Number of stacked RNN layers.
        hidden_dim: Dimension of the RNN hidden state.
    """

    def __init__(self, env, layers=1, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = env.observation_space.shape[0]
        self.out_dimension = env.action_space.n

        self.rnn = nn.RNN(
            self.in_channels, hidden_dim, layers,
            nonlinearity="relu", batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, self.out_dimension)


    def forward(self, history):
        """Compute Q-values from an observation history.

        Args:
            history: Observation sequence (deque or list).

        Returns:
            Q-values tensor of shape '(1, out_dimension)'.
        """
        seq = torch.tensor(
            list(history), requires_grad=False, dtype=torch.float
        ).unsqueeze(0)
        out, _ = self.rnn(seq)
        return self.fc(out[:, -1, :])


class LSTMQFunction(AbstractQFunction):
    """Q-function using an LSTM to process the observation sequence.

    Args:
        env: Gym-compatible environment.
        layers: Number of stacked LSTM layers.
        hidden_dim: Dimension of the LSTM hidden state.
    """

    def __init__(self, env, layers=1, hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = env.observation_space.shape[0]
        self.out_dimension = env.action_space.n

        self.lstm = nn.LSTM(
            self.in_channels, hidden_dim, layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, self.out_dimension)


    def forward(self, history):
        """Compute Q-values from an observation history.

        Args:
            history: Observation sequence (deque or list).

        Returns:
            Q-values tensor of shape '(1, out_dimension)'.
        """
        seq = torch.tensor(
            list(history), requires_grad=False, dtype=torch.float
        ).unsqueeze(0)
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])


class RandomPolicy(nn.Module):
    """Baseline policy that selects actions uniformly at random.

    Returns a one-hot Q-value tensor so that 'argmax' yields the randomly
    chosen action.  Has no learnable parameters — intended for evaluation only.
    """

    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = env.action_space.n

    def reset_state(self):
        """No-op: random policy has no internal state."""

    def forward(self, history):
        """Return one-hot Q-values for a randomly chosen action."""
        action = np.random.randint(0, self.n_actions)
        return torch.eye(self.n_actions)[action].unsqueeze(0)
