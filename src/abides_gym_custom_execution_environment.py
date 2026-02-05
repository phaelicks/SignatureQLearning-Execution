import importlib
from typing import Any, Dict, List, Union, Optional

import gym
import numpy as np

import abides_markets.agents.utils as markets_agent_utils
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_core.generators import ConstantTimeGenerator

from abides_gym.envs.markets_environment import AbidesGymMarketsEnv

class SubGymMarketsCustomExecutionEnv(AbidesGymMarketsEnv):
    """
    Execution v1 environment, it defines a new ABIDES-Gym-markets environment.
    It provides an evironment for a simple algorithmic order execution problem 
    The agent has either an initial inventory of the stocks it tries sell or 
    no initial inventory and tries to acquire a target number of shares. It can do 
    so by sending limit sell and limit buy orders at the first level in the book
    respectively. No market impact is included in this environment. The idea is 
    that an agent in this environment should learn a very specific policy, namely 
    to flatten out all inventory or acquire the desired quantity as soon as 
    possible and keep it that way until the end of trading day.
        
    Arguments:
        - background_config: the handcrafted agents configuration used for the environnement
        - mkt_close: time the market day ends
        - timestep_duration: how long between 2 wake ups of the gym experimental agent
        - starting_cash: cash of the agents at the beginning of the simulation
        - order_fixed_size: size of the limit orders placed by the experimental gym agent
        - state_history_length: length of the raw state buffer
        - market_data_buffer_length: length of the market data buffer
        - first_interval: how long the simulation is run before the first wake up of the gym experimental agent
        - observation_interval: how long the gym experimential agent only observes the market after first_interval 
        - max_inventory: absolute value of maximum inventory the experimental gym agent is allowed to accumulate
        - starting_inventory: inventory units the gym agent starts with at market open
        - running_inventory_reward_dampener: parameter that defines dampening of rewards from speculation
        - terminal_inventory_reward: max terminal reward achievable with zero inventory at market close
        - debug_mode: arguments to change the info dictionnary (lighter version if performance is an issue)
        - background_config_extra_kvargs: dictionary of extra key value  arguments passed to the background config builder function
    
    Market Maker V0:
        - Action Space:
            - LMT BUY of order_fixed_size at best bid price in current book
            - LMT SELL of order_fixed_size at best ask price in current book
            - Do nothing
        - State Space:
            - remaining_time_pct
            - inventory_pct
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
            state_history_length: int = 1,  
            market_data_buffer_length: int = 1,
            first_interval: str = "00:05:00",
            observation_interval: str = "00:01:00",
            order_fixed_size: int = 100,
            max_inventory: int = 1000,
            starting_inventory: Union[int,str] = 0,            
            terminal_inventory_reward: float = 0, 
            terminal_inventory_mode: str = 'quadratic',
            running_inventory_reward_dampener: float = 0.,
            reward_multiplier: Optional[str] = None,
            reward_multiplier_float: Optional[float] = None,
            damp_mode: Optional[str] = None,
            debug_mode: bool = False,
            background_config_extra_kvargs: Dict[str, Any] = {}
    ) -> None: 
        self.background_config: Any = importlib.import_module(
            "abides_markets.configs.{}".format(background_config), package=None
        )
        self.mkt_close: NanosecondTime = str_to_ns(mkt_close)
        self.timestep_duration: NanosecondTime = str_to_ns(timestep_duration)
        self.starting_cash: int = starting_cash
        self.state_history_length: int = state_history_length
        self.market_data_buffer_length: int = market_data_buffer_length
        self.first_interval: NanosecondTime = str_to_ns(first_interval)
        self.observation_interval: NanosecondTime = str_to_ns(observation_interval)
        self.order_fixed_size: int = order_fixed_size
        self.max_inventory: int = max_inventory
        self.starting_inventory: int = starting_inventory
        self.terminal_inventory_reward: float = terminal_inventory_reward
        self.terminal_inventory_mode: str = terminal_inventory_mode        
        self.running_inventory_reward_dampener: float = running_inventory_reward_dampener
        self.reward_multiplier: Optional[str] = reward_multiplier
        self.reward_multiplier_float: Optional[float] = reward_multiplier_float
        self.damp_mode: Optional[str] = damp_mode
        self.debug_mode: bool = debug_mode

        # time the market is open
        self.mkt_open_duration: NanosecondTime = self.mkt_close - str_to_ns("09:30:00")

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
            self.timestep_duration <= self.mkt_open_duration) & (
            self.timestep_duration >= str_to_ns("00:00:00")
            ), "Select authorized timestep_duration"

        assert (type(self.starting_cash) == int) & (
            self.starting_cash >= 0
        ), "Select positive integer value for starting_cash" 

        assert (type(self.state_history_length) == int) & (
            self.state_history_length >= 0
        ), "Select positive integer value for state_history_length"

        assert (type(self.market_data_buffer_length) == int) & (
            self.market_data_buffer_length >= 0
        ), "Select positive integer value for market_data_buffer_length"

        assert (self.first_interval <= self.mkt_open_duration) & (
            self.first_interval >= str_to_ns("00:00:00")
        ), "Select authorized FIRST_INTERVAL delay"

        assert (self.observation_interval <= self.mkt_open_duration - self.first_interval) & (
            self.observation_interval >= str_to_ns("00:00:00")
        ), "Select authorized observation_interval duration"

        assert (type(self.order_fixed_size) == int) & (
            self.order_fixed_size >= 0
        ), "Select positive integer value for order_fixed_size"        

        assert (type(self.max_inventory) == int) & (
            self.max_inventory >= 0
        ), "Select positive integer value for max_inventory"

        correct_int_or_string = (
            (0 <= abs(self.starting_inventory) <= self.max_inventory)
            or self.starting_inventory == "random"
        )
        assert (type(self.starting_inventory) in (int, str)) & (
            correct_int_or_string
        ), "Select positive integer value, smaller than max_inventory, \
            or random for starting_inventory."

        assert (
            self.starting_inventory % self.order_fixed_size == 0
        ), "Select starting_inventory as multiple of order_fixed_size"

        assert (
            type(self.terminal_inventory_reward) == float
        ), "Select float value for terminal_inventory_reward"

        assert (type(self.running_inventory_reward_dampener) == float) & (
            0 <= self.running_inventory_reward_dampener <= 1
        ), "Select positive float value for running_inventory_reward_dampener between 0 and 1"

        assert damp_mode in [
            "asymmetric",
            "symmetric",
            None,
        ], "damp_mode needs to be symmetric, asymmetric or None"

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
            first_interval=self.first_interval, # time delay before first wakeup of gym agent
        )

        # ACTION SPACE
        # LMT BUY of order_fixed_size, 
        # LMT SELL of order_fixed_size, 
        # Do nothing
        self.num_actions: int = 3
        self.action_space: gym.Space = gym.spaces.Discrete(self.num_actions)
        self.do_nothing_action_id: int = self.num_actions - 1            

        # track spread and mid price for action translation
        self.spread: float = 0
        self.current_mid_price: int = 100_000
        self.previous_mid_price: int = 100_000

        # track inventory for MKT order action and reward calculation
        self.previous_inventory: int = self.starting_inventory
        self.current_inventory: int = self.starting_inventory        

        # STATE SPACE
        # [remaining_time_pct, inventory_pct]
        self.num_state_features: int = 2
        
        # create state space "box"
        self.state_highs: np.ndarray = np.array(
            [
                2, # remaining_time_pct
                2, # inventory_pct
            ],
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.state_lows: np.ndarray = np.array(
            [
                -2, # remaining_time_pct
                -2, # inventory_pct
            ],
            dtype=np.float32,
        ).reshape(self.num_state_features, 1)

        self.observation_space: gym.Space = gym.spaces.Box(
            self.state_lows,
            self.state_highs,
            shape=(self.num_state_features, 1),
            dtype=np.float32
        )

        # REWARDS
        self.previous_cash = self.starting_cash


    # UTILITY FUNCTIONS that translate between gym environment and ABIDES simulation

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(
        self, action: int
    ) -> List[Dict[str, Any]]:
        """
        utility function that maps OpenAI action definition (integers) 
        to environnement API action definition (list of dictionaries)
        The action space ranges [0, 1, 2] where:
        - `0` LMT BUY order of order_fixed_size at best bid level
        - `1` LMT SELL order of order_fixed_size at best ask level
        - '2' DO NOTHING

        Note: LMT oder price levels are defined via the current spread to
        handle simulator pathologies such as an empty order book.
        Orders existing in the book are cancelled before a new order is submitted.

        Arguments:
            - action: integer representation of the different actions

        Returns:
            - action_list: list of the corresponding series of action mapped into abides env apis
        """
        
        if action in range(self.num_actions - 1):
            half_spread = self.spread / 2
            bid_price = round(self.current_mid_price - half_spread * 1)
            ask_price = round(self.current_mid_price + half_spread * 1)

            cancel = {"type": "CCL_ALL"} # TODO: check order status, keep existing orders if on correct level
            lmt_buy = {
                "type": "LMT",
                "direction": "BUY",
                "size": self.order_fixed_size,
                "limit_price": bid_price
            }
            lmt_sell = {
                "type": "LMT",
                "direction": "SELL",
                "size": self.order_fixed_size,
                "limit_price": ask_price
            }

            if (action == 0) and (
                self.current_inventory <= self.max_inventory - self.order_fixed_size            
            ): return [cancel, lmt_buy]
            elif (action == 1) and (
                self.current_inventory >= -self.max_inventory + self.order_fixed_size
            ): return [cancel, lmt_sell]
            else: return []          
  
        elif action == self.do_nothing_action_id:
            return []
        else:
            raise ValueError(
                f"Action {action} is not part of the actions support by this environment."
            )       


    @raw_state_to_state_pre_process
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state / observation representation for the market making v0 environnement
        """
                
        # 0)  Preliminary
        # 0) a) compute & save spread for action selection
        bids = raw_state["parsed_mkt_data"]["bids"]
        asks = raw_state["parsed_mkt_data"]["asks"]
        last_transactions = raw_state["parsed_mkt_data"]["last_transaction"]
        
        mid_prices = [
            markets_agent_utils.get_mid_price(b, a, lt)
            for (b, a, lt) in zip(bids, asks, last_transactions)
        ]
        best_bids = [
            bids[0][0] if len(bids) > 0 else mid
            for (bids, mid) in zip(bids, mid_prices)
        ]
        best_asks = [
            asks[0][0] if len(asks) > 0 else mid
            for (asks, mid) in zip(asks, mid_prices)
        ]

        spreads = np.array(best_asks) - np.array(best_bids)
        self.spread = spreads[-1]
        self.previous_mid_price = self.current_mid_price
        self.current_mid_price = mid_prices[-1]
        if len(mid_prices) > 1:
            print(mid_prices[-2] - self.previous_mid_price)

        """
        # TODO: explore the use of moving average of spread instead of just current spread
        # moving average of spreads:
        print("spread moving average: {}".format(spreads.mean()))
        print("current spread: {}".format(self.spread))
        """

        # 1) Timing
        mkt_open = raw_state["internal_data"]["mkt_open"][-1]
        mkt_close = raw_state["internal_data"]["mkt_close"][-1]
        current_time = raw_state["internal_data"]["current_time"][-1]
        assert (
            current_time >= mkt_open + self.first_interval
        ), "Agent has woken up earlier than its first interval"
        elapsed_time = current_time - mkt_open - self.first_interval
        total_time = mkt_close - mkt_open - self.first_interval
        # percentage time advancement
        time_pct = (total_time - elapsed_time) / total_time

        # 2) Inventory
        holdings = raw_state["internal_data"]["holdings"]
        self.previous_inventory = self.current_inventory # save for reward calculation
        self.current_inventory = self.starting_inventory + holdings[-1] # save for reward calculation
        inventory_pct = self.current_inventory / self.max_inventory

        # log custom metrics to tracker
        # TODO: implement custom metrics tracker

        # computed state
        computed_state = np.array(
            [
                time_pct,
                inventory_pct,
            ], dtype=np.float64
        )

        return computed_state.reshape(self.num_state_features, 1)

    @raw_state_pre_process
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        Method that transforms a raw state into the reward obtained during the step.

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: immediate reward computed at each step  for the execution v0 environnement
        """
        
        # We define the (running) reward as the change in inventory value due to midprice fluctuations. 
        # Additionally, this reward can be dampened either symmetrically (positive and negative rewards)
        # or asymmetrically (only positive rewards), see (Spooner et al. (2018)).        

        # 1) change in inventory value
        mid_price_change = (self.current_mid_price - self.previous_mid_price) / 100 # dollar terms
        inventory_pct = self.current_inventory / self.max_inventory

        if self.reward_multiplier == 'flat':
            offset = (0.5 * self.order_fixed_size) / self.max_inventory
            #inventory_pct = np.sign(inventory_pct) * max(abs(inventory_pct) - offset, 0)
            # linear flat both sides
            inventory_reward = - max(abs(inventory_pct) - offset, 0.) * abs(mid_price_change)

        
        if self.reward_multiplier == 'quadratic':
            #inventory_pct = np.sign(inventory_pct) * inventory_pct ** 2
            
            # quadratic both sides
            inventory_reward = - (inventory_pct ** 2) * abs(mid_price_change)

        if self.reward_multiplier == 'quadratic_flat':            
            # quadratic both sides with flat part
            offset = (0.5 * self.order_fixed_size) / self.max_inventory
            inventory_reward = - (max(abs(inventory_pct) - offset, 0.) ** 2) * abs(mid_price_change)

        if self.reward_multiplier == 'quadratic_positive':            
            # quadratic both sides with positive part
            offset = (0.5 * self.order_fixed_size) / self.max_inventory
            inventory_reward = - (inventory_pct ** 2 - offset ** 2) * abs(mid_price_change)
    

        if self.reward_multiplier_float is not None:
            inventory_reward *= self.reward_multiplier_float        

        # damp inventory reward 
        if self.damp_mode == "symmetric":
            inventory_reward *= (1 - self.running_inventory_reward_dampener)
        elif self.damp_mode == "asymmetric":
            inventory_reward -= max(
                0.,
                self.running_inventory_reward_dampener * inventory_reward
            )
        self.inventory_reward = inventory_reward #/ self.order_fixed_size
        
        # TODO: normalize for order size and max inventory?
        #reward = pnl / self.order_fixed_size + inventory_change / self.max_inventory
        
        return inventory_reward

    @raw_state_pre_process
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        Method that transforms a raw state into the final step reward update.

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode for the daily investor v0 environnement
        """

        # 1) inventory pct
        inventory = self.starting_inventory + raw_state["internal_data"]["holdings"]
        inventory_pct = inventory / self.max_inventory # self.starting_inventory

        #### WRONG TERMINAL REWARD - leads to interesting behaviour
        #update_reward = - self.terminal_inventory_reward * (inventory_pct ** 2)
        ####
        update_reward = 0
        offset = 0.5 * self.order_fixed_size / self.max_inventory
        
        if self.terminal_inventory_mode == 'quadratic':
            # quadratic reward / penalty depending on sign of terminal_inventory_reward
            if self.reward_multiplier == 'quadratic':
                update_reward = (
                    self.terminal_inventory_reward * (inventory_pct ** 2) # penalty
                    if self.terminal_inventory_reward < 0 else
                    self.terminal_inventory_reward * (1 - inventory_pct ** 2) # reward
                )
            elif self.reward_multiplier == 'quadratic_flat':
                update_reward = (
                    self.terminal_inventory_reward * (max(abs(inventory_pct) - offset, 0.) ** 2) # penalty
                    if self.terminal_inventory_reward < 0 else
                    self.terminal_inventory_reward * (1 - (max(abs(inventory_pct) - offset, 0.) ** 2)) # reward
                )
            elif self.reward_multiplier == 'quadratic_positive':
                update_reward = (
                    self.terminal_inventory_reward * (inventory_pct ** 2 - offset ** 2) # penalty
                    if self.terminal_inventory_reward < 0 else
                    self.terminal_inventory_reward * (1 - (inventory_pct ** 2 - offset ** 2)) # reward
                )
        elif self.terminal_inventory_mode == 'linear':
            # linear reward / penalty depending on sign of terminal_inventory_reward
            update_reward = (
                self.terminal_inventory_reward * abs(inventory_pct) # penalty
                if self.terminal_inventory_reward < 0 else
                self.terminal_inventory_reward * (1 - abs(inventory_pct)) # reward
            )
        elif self.terminal_inventory_mode == 'flat':
            # linear flat reward / penalty
            offset = 0.5 * self.order_fixed_size / self.max_inventory
            new_inventory_pct = max(abs(inventory_pct) - offset, 0.)
            update_reward = (
                self.terminal_inventory_reward * new_inventory_pct # penalty
                if self.terminal_inventory_reward < 0 else
                self.terminal_inventory_reward * (1 - new_inventory_pct) # reward
            )
        else:
            raise ValueError("Select quadratic, linear or flat for terminal_inventory_mode")
        
        return update_reward #/ self.order_fixed_size

    @raw_state_pre_process
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> bool:
        """
        method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not  for the execution v0 environnement
        """
        # episode can stop because market closes (or because some condition is met)
        # here no other condition is used (such as running out of cash)
        return

    @raw_state_pre_process
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step for the daily investor v0 environnement
        """
        # Agent cannot use this info for taking decision
        # only for debugging

        # 1) Last Known Market Transaction Price
        last_transaction = raw_state["parsed_mkt_data"]["last_transaction"]

        # 2) Last Known best bid
        bids = raw_state["parsed_mkt_data"]["bids"]
        best_bid = bids[0][0] if len(bids) > 0 else last_transaction

        # 3) Last Known best ask
        asks = raw_state["parsed_mkt_data"]["asks"]
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        # 4) Available Cash
        cash = raw_state["internal_data"]["cash"]

        # 5) Current Time
        current_time = raw_state["internal_data"]["current_time"]

        # 6) Holdings
        inventory = self.starting_inventory + raw_state["internal_data"]["holdings"]

        # 7) Spread
        spread = best_ask - best_bid

        # 8) OrderBook features
        orderbook = {
            "asks": {"price": {}, "volume": {}},
            "bids": {"price": {}, "volume": {}},
        }

        for book, book_name in [(bids, "bids"), (asks, "asks")]:
            for level in [0, 1, 2]:
                price, volume = markets_agent_utils.get_val(book, level)
                orderbook[book_name]["price"][level] = np.array([price]).reshape(-1)
                orderbook[book_name]["volume"][level] = np.array([volume]).reshape(-1)

        # 9) order_status
        order_status = raw_state["internal_data"]["order_status"]

        # 10) mkt_open
        mkt_open = raw_state["internal_data"]["mkt_open"]

        # 11) mkt_close
        mkt_close = raw_state["internal_data"]["mkt_close"]

        # 12) last vals
        last_bid = markets_agent_utils.get_last_val(bids, last_transaction)
        last_ask = markets_agent_utils.get_last_val(asks, last_transaction)

        # 13) spreads
        wide_spread = last_ask - last_bid
        ask_spread = last_ask - best_ask
        bid_spread = best_bid - last_bid

        # 14) compute the marked to market
        marked_to_market = cash + inventory * last_transaction

        # 15) pnl = self.pnl
        cash = raw_state["internal_data"]["cash"]
        pnl = (cash - self.previous_cash) / 100 # in dollar terms
        self.pnl = pnl / self.order_fixed_size
        self.previous_cash = cash
        
        # 16) inventory_reward = self.inventory_reward
        # 17) reward = self.pnl + self.inventory_reward
        # 18) mid_price = self.mid_price

        if self.debug_mode == True: # info returned in debug mode
            return {
                "last_transaction": last_transaction,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "bids": bids,
                "asks": asks,
                "cash": cash,
                "current_time": current_time,
                "inventory": inventory,
                "orderbook": orderbook,
                "order_status": order_status,
                "mkt_open": mkt_open,
                "mkt_close": mkt_close,
                "last_bid": last_bid,
                "last_ask": last_ask,
                "wide_spread": wide_spread,
                "ask_spread": ask_spread,
                "bid_spread": bid_spread,
                "marked_to_market": marked_to_market,
                "pnl": self.pnl,
                "inventory_reward": self.inventory_reward,
                "reward": self.pnl + self.inventory_reward,
                "mid_price": self.current_mid_price,
            }
        else: # info always returned
            return {
                "cash": cash,
                "pnl": self.pnl,
                "inventory": inventory,
                "inventory_reward": self.inventory_reward,
                "mid_price": self.current_mid_price,
            }   

    def close(self) -> None:
        """
        Closes the environment and performs necassary clean up such as setting internal
        variables to initial value for next reset call.
        """    
        # set internal variables to default values for next episode
        self.current_mid_price = 100_000
        self.previous_mid_price = 100_000 
        self.current_inventory = self.starting_inventory
        self.previous_inventory = self.starting_inventory
        self.previous_cash = self.starting_cash

        return





    

