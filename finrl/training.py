import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from config import TRAIN_CSV
import sys
import gc
# from custom_env2 import CustomStockTradingEnv
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np



def setup_environment(agent : str):
    """
    This function loads the train dataset and sets up the environment for training.

    Args:
        None
        
    """ 
    # AGENT_ENV_OVERRIDES = {
    #     "a2c": {"reward_scaling": 1e-9, "hmax": 500},
    #     "ppo": {"reward_scaling": 1e-6, "hmax": 500},
    #     "sac": {"reward_scaling": 1e-4, "hmax": 500},
    #     "td3": {"reward_scaling": 1e-2, "hmax": 500},
    #     }
    AGENT_ENV_OVERRIDES = {
    "a2c": {"reward_scaling": 1e-4, "hmax": 500},
    "ppo": {"reward_scaling": 1e-2, "hmax": 500},
    "sac": {"reward_scaling": 1e-3, "hmax": 500},
    "td3": {"reward_scaling": 1e-2, "hmax": 500},
    }
    # Override the environment parameters based on the agent type

    try: 
        # Load train data
        train = pd.read_csv(TRAIN_CSV)
        train = train.set_index(train.columns[0])
        train.index.names = ['']

        # Environment setup
        stock_dimension = len(train.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        buy_cost_list = sell_cost_list = [0.001] * stock_dimension  
        num_stock_shares = [0] * stock_dimension

        override = AGENT_ENV_OVERRIDES.get(agent.lower(), {})

        env_kwargs = {
            "hmax": 500,                         # Max shares to trade per step
            "initial_amount": 100000,            # Lower for intraday
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": override.get("reward_scaling", 1e-4)  # Smaller for hourly granularity
        }

        e_train_gym = StockTradingEnv(df = train, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        
        return env_train
    
    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        sys.exit(1)


def train_a2c():
    """
    This function trains the A2C model using historical stock price data.
    This function is called only if it is necessary to retrain the model.

    Args:
        None
        
    """ 
    try:
        env_train_a2c = setup_environment("a2c")
        print("Environment for A2C agent set up successfully.")

        A2C_PARAMS = {
            "n_steps": 64,                     # Smaller n_steps for hourly data
            "ent_coef": 4.053861348529222e-05,                 # Entropy coefficient
            "learning_rate": 6.521490192554121e-05,           # Learning rate
            "gamma": 0.9332012643665162                    # Discount factor 
        }


        # Initialize agent
        agent_a2c = DRLAgent(env=env_train_a2c)
        model_a2c = agent_a2c.get_model("a2c", model_kwargs=A2C_PARAMS)

        # Train the agent
        trained_a2c = agent_a2c.train_model(
            model=model_a2c,
            tb_log_name='a2c_hourly',
            total_timesteps=150000   # Slightly more steps due to smaller interval
        )


        trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
        
        tmp_path = RESULTS_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_a2c.set_logger(new_logger_a2c)
        print("A2C model logger configured successfully.")
        # Clean up
        import gc
        del agent_a2c, model_a2c, trained_a2c
        gc.collect()
    
    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        sys.exit(1)


def train_sac():
    """
    This function trains the SAC model using historical stock price data.
    This function is called only if it is necessary to retrain the model.

    Args:
        None
        
    """ 
    try:
        env_train_sac = setup_environment("sac")
        print("Environment for SAC agent set up successfully.")
        
        # Initialize agent
        agent_sac = DRLAgent(env=env_train_sac)

        SAC_PARAMS = {
            "batch_size": 128,
            "buffer_size": 100_000,
            "learning_rate": 1e-4,  
            "learning_starts": 5000,
            "ent_coef": 0.005,  # fixed, not auto
            "train_freq": 1,
            "gradient_steps": 1,
            "tau": 0.005,
            "gamma": 0.99
        }
        

        POLICY_KWARGS = {
            "net_arch": [256, 256],
            "log_std_init": -1
        }
        
        model_sac = agent_sac.get_model(
            "sac", 
            model_kwargs = SAC_PARAMS,
            policy_kwargs=POLICY_KWARGS
        )
        
        tmp_path = RESULTS_DIR + '/sac'
        new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_sac.set_logger(new_logger_sac)

        # Train the agent
        trained_sac = agent_sac.train_model(
            model=model_sac,
            tb_log_name='sac_hourly',
            total_timesteps=300000   
        )

        trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
        del agent_sac, model_sac, trained_sac
        gc.collect()
    
    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        sys.exit(1)

def train_ppo():
    """
    This function trains the PPO model using historical stock price data.
    This function is called only if it is necessary to retrain the model.

    Args:
        None
        
    """ 
    try:
        env_train_ppo = setup_environment("ppo")
        print("Environment for PPO agent set up successfully.")
        
        # Initialize agent
        agent_ppo = DRLAgent(env=env_train_ppo)
        
        PPO_PARAMS = {
            "n_steps": 128,  # Smaller n_steps for hourly data
            "learning_rate": 1.3575277728743006e-05,
            "ent_coef": 2.9425153731115703e-06,
            "gamma": 0.9951611415999558
        }

        model_ppo = agent_ppo.get_model("ppo", model_kwargs=PPO_PARAMS)
        tmp_path = RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)
        

        # Train the agent
        trained_ppo = agent_ppo.train_model(
            model=model_ppo,
            tb_log_name='ppo_hourly',
            total_timesteps=300000   
        )

        trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")
        print("PPO training complete and model saved.")
        
        del agent_ppo, model_ppo, trained_ppo
        gc.collect()
    
    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        sys.exit(1)

def train_td3():
    """
    This function trains the TD3 model using historical stock price data.
    This function is called only if it is necessary to retrain the model.

    Args:
        None
        
    """ 
    try:
        env_train_td3 = setup_environment("td3")
        print("TD3 environment configured and training started …")
        # Initialize agent
        agent_td3 = DRLAgent(env=env_train_td3)

        TD3_PARAMS = {
            "buffer_size":     300_000,
            "batch_size":      256,
            "learning_rate":   1e-4,  # Increased learning rate
            "gamma":           0.99,  # More conservative discount
            "tau":             0.005,
            "policy_delay":    2,
            "train_freq":      (1, "step"),  # Ensure regular updates
            "gradient_steps":  1,
            "target_policy_noise": 0.2,
            "target_noise_clip":  0.5,
        }
        # Action noise for exploration
        n_actions = env_train_td3.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model_td3 = agent_td3.get_model("td3", model_kwargs=TD3_PARAMS)
        model_td3.action_noise = action_noise
        tmp_path = RESULTS_DIR + "/td3"
        model_td3.set_logger(configure(tmp_path, ["stdout", "csv", "tensorboard"]))

        # Train the agent
        trained_td3 = agent_td3.train_model(
            model=model_td3,
            tb_log_name='td3_hourly',
            total_timesteps=300000  # Increased timesteps for more training   
        )

        trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")
        print("TD3 training complete and model saved.")
        del agent_td3, model_td3, trained_td3
        gc.collect()

    
    except Exception as e:
        print(f"[Error in finrl training.py] -> {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Ensemble trainer -------------------------------------
# ---------------------------------------------------------------------------

def train_all():
    """Train **all** supported agents sequentially for ensemble setups."""
    print("🏁 Starting full ensemble training A2C, PPO, SAC, TD3 …")
    train_a2c()
    train_ppo()
    train_sac()
    train_td3()
    print("✅ All agents trained and saved under", TRAINED_MODEL_DIR)

# ---------------------------------------------------------------------------