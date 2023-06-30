import importlib
from typing import Any, Dict, List

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from abides_gym.envs.markets_environment import AbidesGymMarketsEnv

class SubGymMarketsMarketMakingEnv_v0(AbidesGymMarketsEnv):
    """
    Market Making v0 environment, it defines a new ABIDES-Gym-markets environment.
    It provides an evironment for the problem of a market maker trying to maximize its 
    return by continuously posting (limit) buy and (limit) sell orders to capture the 
    spread while at the same time keeping its inventory low. The market maker starts 
    the day with cash but no position, then continously chooses bid and ask levels at 
    which to post limit orders.  At the end of the day all remaining inventory is liquidated.
    
    Arguments:
        - background_config: the handcrafted agents configuration used for the environnement
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wakes up of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the limit orders placed by the experimental gym agent
        - mkt_order_alpha: proportion of inventory for market orders placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - last_intervall: how long before market close the gym experimental agent stops trading
        TODO: implement functionality to stop at mkt_clos - last_intervall
        - max_inventory: absolute value of maximum inventory the experimental gym agent is allowed to accumulate
        - leftover_inventory_reward: a constand penalty per unit of inventory at market close
        - reward_mode: can use a dense of sparse reward formulation
        - done_ratio: ratio (mark2market_t/starting_cash) that defines when an episode is done (if agent has lost too much mark to market value)
        - debug_mode: arguments to change the info dictionnary (lighter version if performance is an issue)
        - background_config_extra_kvargs: dictionary of extra key value  arguments passed to the background config builder function
    
    Market Maker V0:
        - Action Space:
            - [LMT BUY, LMT SELL] combinations of order_fixed_size: 
                {[], [] [], [], []} 
            - MKT BUY of mkt_order_alpha * inventory_t
            - MKT SELL of mkt_order_alpha * inventory_t
            - Do nothing
        - State Space:
            - remaining_time_pct
            - inventory_pct
            - mid_price
            - lagged_mid_price
            - imbalance_5
            – market_spread
    """

    # Decorator for functions to ignore buffering in market data and generl raw state
    raw_state_pre_process = markets_agent_utils.ignore_buffers_decorator
    raw_state_to_state_pre_process = (
        markets_agent_utils.ignore_mkt_data_buffer_decorator
    )

    def __init__(
            self,       
            background_config: Any = "rmsc04",
            mkt_close: str = "16:00:00",
            timestep_duration: str = "10s",
            starting_cash: int = 100_000,
            order_fixed_size: int = 10,
            mkt_order_alpha: float = 0.1,
            state_history_length: int = 2,  
            market_data_buffer_length: int = 5,
            first_interval: str = "00:15:00",
            last_interval: str = "00:00:00",
            max_inventory: int = 1000,
            remaining_inventory_reward: int = -100, 
            reward_mode: str = "dense",
            done_ratio: float = 0.2,
            debug_mode: bool = False,
            background_config_extra_kvargs: Dict[str, Any] = {}
    ) -> None: 
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.order_fixed_size: int = order_fixed_size
        self.mkt_order_alpha: float = mkt_order_alpha
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.last_interval: NanosecondTime = str_to_ns(last_interval)
        self.max_inventory: int = max_inventory
        self.remaining_inventory_reward: int = remaining_inventory_reward
        self.done_ratio: float = done_ratio
        self.debug_mode: bool = debug_mode

        # time the market is open
        self.mkt_open_duration: NanosecondTime = self.mkt_close - str_to_ns("09:30:00")

        # marked_to_market limit to STOP the epsidoe
        self.down_done_condition: float = self.done_ratio * starting_cash

        # CHECK PROPERTIES
        assert background_config in [
            "rmsc03",
            "rmsc04",
            "smc_01",
        ], "Select rmsc03, rmsc04 or smc_01 as config"

        assert (self.mkt_close <= str_to_ns("16:00:00")) & (
            self.mkt_close >= str_to_ns("09:30:00")
        ), "Select authorized market hours"

        assert (
            self.timestep_duration <= self.mkt_open_duration - self.last_interval) & (
            self.timestep_duration >= str_to_ns("00:00:00")
            ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
            self.starting_cash >= 0
        ), "Select positive integer value for starting_cash" 

        assert (type(self.order_fixed_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.mkt_order_alpha) == float) & (
            0 <= self.mkt_order_alpha <= 1
        ), "Select positive float value for mkt_order_alpha between 0 and 1"

        assert (type(self.state_history_length) == int) & (
            self.state_history_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (type(self.market_data_buffer_length) == int) & (
            self.market_data_buffer_length >= 0
        ), "Select positive integer value for order_fixed_size"

        assert (self.first_interval <= self.mkt_open_duration) & (
            self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.last_interval >= str_to_ns("00:00:00")) & (
            self.last_interval <= self.mkt_open_duration
        ), "Select authorized LAST_INTERVAL stop before market close"

        assert (
            self.first_interval + self.last_interval <= self.mkt_open_duration
        ), "Select authorized FIRST_INTERVAL and LAST_INTERVAL combination"    

        assert (type(self.max_inventory) == int) & (
            self.max_inventory >= 0
        ), "Select positive integer value for max_inventory"

        assert (
            type(self.remaining_inventory_reward) == int
        ), "Select integer value for remaining_inventory_reward"

        assert (type(self.done_ratio) == float) & (
            0 <= self.done_ratio <= 1
        ), "Select positive float value for done_ration between 0 and 1"

        assert self.debug_mode in [
            True,
            False,
        ], "debug_mode needs to be True or False"                


        # BACKGROUND CONFIG
        background_config_args = {"end_time": self.mkt_close}
        background_config_args.update(background_config_extra_kvargs)
        
        # INIT
        super().__init__(
            background_config_pair=(
                self.background_config.build_config,
                background_config_args,
            ),
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=self.timestep_duration
            ),
            starting_cash=self.starting_cash,
            state_buffer_length=self.state_history_length,
            market_data_buffer_length=self.market_data_buffer_length,
            first_interval=self.first_interval,
        )

        # ACTION SPACE
        # 9 LMT spreads order_fixed_size, 
        # MKT inventory * mkt_order_alpha
        # Do nothing
        self.num_actions: int = 11
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)

        # STATE SPACE
        # [remaining_time_pct, inventory_pct, mid_price, 
        #   lagged_mid_price, imbalance_5, market_spread]
        self.num_state_features: int = 6
        
        # create state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                2, # remaining_time_pct
                2, # inventory_pct
                np.finfo(np.float).max, # mid_price
                np.finfo(np.float).max, # lagged_mid_price
                1, # imbalance_5
                np.finfo(np.float).max # market_spread
            ]
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                -2, # remaining_time_pct
                -2, # inventory_pct
                np.finfo(np.float).min, # mid_price
                np.finfo(np.float).min, # lagged_mid_price
                -1, # imbalance_5
                np.finfo(np.float).min # market_spread
            ]
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32
        )
        self.previous_marked_to_market: int = self.starting_cash


    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        return
        

    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        return

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        return

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        return

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        return

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        return





    

