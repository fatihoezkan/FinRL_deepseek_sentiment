import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from config import TRAIN_CSV
import sys


def setup_environment(agent : str):
    """
    This function loads the train dataset and sets up the environment for training.

    Args:
        None
        
    """ 
    AGENT_ENV_OVERRIDES = {
        "a2c": {"reward_scaling": 1e-4, "hmax": 500},
        "ppo": {"reward_scaling": 1e-4, "hmax": 500},
        "sac": {"reward_scaling": 1e-10, "hmax": 1000},
        "td3": {"reward_scaling": 1e-10, "hmax": 1000},
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
        
        # Initialize agent
        agent_a2c = DRLAgent(env=env_train_a2c)
        model_a2c = agent_a2c.get_model("a2c")

        # Train the agent
        trained_a2c = agent_a2c.train_model(
            model=model_a2c,
            tb_log_name='a2c_hourly',
            total_timesteps=150000   # Slightly more steps due to smaller interval
        )

        trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
    
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
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
        
        model_sac = agent_sac.get_model("sac", model_kwargs = SAC_PARAMS)
        
        tmp_path = RESULTS_DIR + '/sac'
        new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_sac.set_logger(new_logger_sac)

        # Train the agent
        trained_sac = agent_sac.train_model(
            model=model_sac,
            tb_log_name='sac_hourly',
            total_timesteps=150000   
        )

        trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
        
    
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
            "n_steps": 2048,
            "batch_size": 128,
            "learning_rate": 0.005,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        }

        model_ppo = agent_ppo.get_model("ppo", model_kwargs=PPO_PARAMS)
        tmp_path = RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)
        

        # Train the agent
        trained_ppo = agent_ppo.train_model(
            model=model_ppo,
            tb_log_name='ppo_hourly',
            total_timesteps=150000   
        )

        trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")
        print("PPO training complete and model saved.")
    
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
        
        # Initialize agent
        agent_td3 = DRLAgent(env=env_train_td3)

        TD3_PARAMS = {
            "buffer_size":     300_000,
            "batch_size":      256,
            "learning_rate":   1e-4,
            "gamma":           0.99,
            "tau":             0.005,   # soft update
            "policy_delay":    2,       # the “delayed” part
            "target_policy_noise": 0.1,
            "target_noise_clip":  0.3,
        }

        model_td3 = agent_td3.get_model("td3", model_kwargs=TD3_PARAMS)
        tmp_path = RESULTS_DIR + "/td3"
        model_td3.set_logger(configure(tmp_path, ["stdout", "csv", "tensorboard"]))


        # Train the agent
        trained_td3 = agent_td3.train_model(
            model=model_td3,
            tb_log_name='td3_hourly',
            total_timesteps=150000   
        )

        trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")
        print("TD3 training complete and model saved.")

    
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