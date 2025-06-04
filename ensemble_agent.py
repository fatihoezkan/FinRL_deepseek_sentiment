from finrl.agents.stablebaselines3.models import DRLEnsembleAgent
from stable_baselines3 import A2C, PPO, SAC, TD3
import pandas as pd

# Load dataset
data = pd.read_csv('train_data.csv', index_col=None)
risk_scores = pd.read_csv('aggregated_risk_scores.csv')

#normalize the risk scores make it close to turbulance 
import pandas as pd

import pandas as pd

# Load data
data = pd.read_csv("train_data.csv")
risk_scores = pd.read_csv("aggregated_risk_scores.csv")

# Prepare risk scores
risk_scores = risk_scores.rename(columns={"datetime": "date", "avg_weighted_score": "risk_score"})
risk_scores['date'] = pd.to_datetime(risk_scores['date'], utc=True)
data['date'] = pd.to_datetime(data['date'], utc=True)

# Normalize and smooth risk scores
min_score = risk_scores['risk_score'].min()
max_score = risk_scores['risk_score'].max()
risk_scores['turbulence'] = ((risk_scores['risk_score'] - min_score) / (max_score - min_score)) * 100
risk_scores['turbulence'] = risk_scores['turbulence'].rolling(window=5, min_periods=1).mean()

risk_scores


#import numpy as np
import numpy as np

# Assuming 'data' has multiple tickers and 'close' prices
def calculate_price_turbulence(df):
    df = df.sort_values(['date', 'tic'])
    unique_dates = df['date'].unique()
    price_matrix = df.pivot(index='date', columns='tic', values='close').fillna(method='ffill')
    returns = price_matrix.pct_change().dropna()

    turbulence_index = []
    for i in range(252, len(returns)):
        current_return = returns.iloc[i]
        hist_returns = returns.iloc[i-252:i]
        cov_matrix = hist_returns.cov()
        mean_return = hist_returns.mean()

        diff = current_return - mean_return
        turbulence = diff.T @ np.linalg.pinv(cov_matrix) @ diff
        turbulence_index.append({'date': returns.index[i], 'turbulence': turbulence})

    return pd.DataFrame(turbulence_index)


calculate_price_turbulence(data)

## add tubulance to data 
data = data.merge(risk_scores[['date', 'turbulence']], on='date', how='left')
data = data.merge(calculate_price_turbulence(data)[['date', 'turbulence']], on='date', how='left')
data['turbulence'] = data['turbulence_x'].fillna(data['turbulence_y'])
data = data.drop(columns=['turbulence_x', 'turbulence_y'])

data
# how many nan value for turbulence
print("Number of NaN values in turbulence:", data['turbulence'].isna().sum())

# fill NaN values in turbulence with 0
data['turbulence'] = data['turbulence'].fillna(0)





# Drop any unnecessary column
data = data.drop(columns=['Unnamed: 0'], errors='ignore')
data = data.sort_values("date").reset_index(drop=True)

# Use this turbulence in your tech indicator list
tech_indicator_list = [
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
    "dx_30", "close_30_sma", "close_60_sma", "turbulence",
]

data


ensemble_agent = DRLEnsembleAgent(
    df=data,
    train_period=("2023-10-02", "2024-12-15"),
    val_test_period=("2024-12-15", "2025-01-27"),
    rebalance_window=72, 
    validation_window=24,
    stock_dim=1,
    hmax=500,
    initial_amount=1_000_000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
    reward_scaling=1e-4,
    state_space= 12, # 1 * (len(tech_indicator_list) + 2),
    action_space=1,
    tech_indicator_list=tech_indicator_list,
    print_verbosity=1,
)

summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs={
        "n_steps": 64,
        "ent_coef": 1.6350519013460405e-07,
        "learning_rate": 0.00011104357457415143,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "gamma": 0.9840885093304689
    },
    PPO_model_kwargs={
        "n_steps": 128,
        "ent_coef": 0.01,
        "learning_rate": 0.00024748430709408656,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "n_epochs": 10,
        "batch_size": 64,        
        "gamma": 0.9059127805773255,
        "ent_coef": 0.002497467173355295,
    
    },
    DDPG_model_kwargs={
    "buffer_size": 10_000,
    "learning_rate": 1e-5,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.95,
    "learning_starts": 1000
    },
    SAC_model_kwargs={
        "buffer_size": 1_000_000,
        "learning_rate": 3.105335814861634e-05,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.9808145812362419,
        "learning_starts": 1000,
        "ent_coef": 0.00023474977863315757,
        "n_steps": 128
    },
    TD3_model_kwargs={
        "buffer_size": 100000,
        "learning_rate": 0.00044206219258000827,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.9492492153893055,
        "learning_starts": 1000,
        "ent_coef": 0.0006077583747534099,
        "n_steps": 128
    },
    timesteps_dict={
        "a2c": 150_000,
        "ppo": 150_000,
        "ddpg": 150_000,
        "sac": 150_000,
        "td3": 150_000
    }
)