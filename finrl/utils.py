import pandas as pd
import numpy as np
import yfinance as yf
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, SAC
from pypfopt.efficient_frontier import EfficientFrontier
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from config import TRADE_CSV, AGGREGATED_RISK_SCORE
from custom_env import RiskAwareStockTradingEnv


"""
utils.py contains all the necessary functions needed in inference.py    
""" 

def load_trade():
    trade = pd.read_csv(TRADE_CSV)
    trade = trade.set_index(trade.columns[0])
    trade.index.names = ['']
    
    return trade


def load_trained_a2c():
    trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
    
    return trained_a2c


def load_trained_sac():
    trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")
    
    return trained_sac


def load_aggregated_risk_score(trade):
    sentiment_df = pd.read_csv(AGGREGATED_RISK_SCORE)
        
    # Rename columns for consistency
    sentiment_df = sentiment_df.rename(columns={"datetime": "date", "avg_weighted_score": "risk_score"})

    # Convert to datetime and localize to UTC
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize('UTC')

    # Merge with UTC-aligned trade['date']
    trade_copy = trade.copy()
    trade_copy['date'] = pd.to_datetime(trade_copy['date'], utc=True)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)
    trade_sentiment = pd.merge(trade_copy, sentiment_df, on='date', how='left')

    # Fill missing risk scores with 0
    trade_sentiment['risk_score'] = trade_sentiment['risk_score'].fillna(0)

    return trade_sentiment


def predict_agent_1(trade, trained_model):
    # Setup the environment
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    
    env_kwargs = {
    "hmax": 500,
    "initial_amount": 1000000,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.001] * stock_dimension,
    "sell_cost_pct": [0.001] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-10,
    }

    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
    env_trade, _ = e_trade_gym.get_sb_env()

    df_account_value_agent1, df_actions_agent1 = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym
    )
    
    return df_account_value_agent1, df_actions_agent1


def predict_agent_2(trade, trained_model, trade_sentiment):
    # Setup the environment
    stock_dimension_agent2 = len(trade.tic.unique())
    state_space_agent2 = 1 + 2 * stock_dimension_agent2 + len(INDICATORS) * stock_dimension_agent2

    env_kwargs_agent2 = {
    "hmax": 500,
    "initial_amount": 1000000,
    "num_stock_shares": [0] * stock_dimension_agent2,
    "buy_cost_pct": [0.001] * stock_dimension_agent2,
    "sell_cost_pct": [0.001] * stock_dimension_agent2,
    "state_space": state_space_agent2,
    "stock_dim": stock_dimension_agent2,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension_agent2,
    "reward_scaling": 1e-10,
    }

    # Initialize custom environment with trade_sentiment data
    e_trade_sentiment = RiskAwareStockTradingEnv(df=trade_sentiment, **env_kwargs_agent2)

    env_trade_sentiment, _ = e_trade_sentiment.get_sb_env()
    
    df_account_value_agent2, df_actions_agent2 = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_sentiment
    )

    return df_account_value_agent2, df_actions_agent2


def process_df_for_mvo(df):
    df = df.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]
    df['date'] = pd.to_datetime(df['date'], utc=True)
    tickers = df['tic'].unique()
    mvo = pd.DataFrame(columns=tickers)

    for i in range(df.shape[0] // len(tickers)):
        temp = df.iloc[i * len(tickers):(i + 1) * len(tickers)]
        date = temp['date'].iloc[0]
        mvo.loc[date] = temp['close'].values

    mvo.index = pd.to_datetime(mvo.index, utc=True)
    
    return mvo


def calculate_mvo(trade):
    StockData = process_df_for_mvo(trade)
    arStockPrices = StockData.to_numpy()
    rows, cols = arStockPrices.shape
    
    return StockData, arStockPrices, rows, cols
 
    
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
            
    return StockReturn


def calculate_mean_cov(arStockPrices, rows, cols):
    arReturns = StockReturnsComputing(arStockPrices, rows, cols)
    meanReturns = np.mean(arReturns, axis=0).reshape(-1)
    covReturns = np.cov(arReturns, rowvar=False) + np.eye(cols) * 1e-6

    return meanReturns, covReturns


def calculate_efficient_frontier(meanReturns, covReturns, trade, StockData):
    stock_dimension = len(trade.tic.unique())
    ef = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 1))
    weights = ef.min_volatility() if stock_dimension == 1 else ef.max_sharpe(solver="SCS")

    cleaned_weights_mean = ef.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])

    LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

    TradeData = process_df_for_mvo(trade)
    TradeData.to_numpy()

    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

    return MVO_result


def get_djia_index(trade):
    TRADE_START_DATE = trade["date"].iloc[0]
    TRADE_END_DATE = trade["date"].iloc[-1]

    TRADE_START_DATE = pd.to_datetime(TRADE_START_DATE)
    TRADE_END_DATE = pd.to_datetime(TRADE_END_DATE)

    trade_start_date = TRADE_START_DATE.strftime("%Y-%m-%d")
    trade_end_date = TRADE_END_DATE.strftime("%Y-%m-%d")

    # Download DJIA hourly data between TRADE_START_DATE and TRADE_END_DATE
    df_dji = yf.download(
        tickers="^DJI",  # DJIA Index ticker
        start=trade_start_date,
        end=trade_end_date,
        interval="1h",
    )

    # Reset index and rename columns to match desired format
    df_dji = df_dji.reset_index()
    df_dji = df_dji.rename(columns={'Datetime': 'date'})
    df_dji['date'] = pd.to_datetime(df_dji['date'], utc=True) 
    df_dji['tic'] = '^DJI'

    # Reorder and rename columns to lowercase (to match NVDA format)
    df_dji = df_dji.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Add a column for the day of the week (0 = Monday, 6 = Sunday)
    df_dji['day'] = df_dji['date'].dt.dayofweek

    # Filter out weekends (market is closed)
    df_dji = df_dji[df_dji['day'] < 5]  # Only weekdays
    
    df_dji = df_dji[['date','close']]
    fst_day = df_dji['close'].iloc[0]
    dji = pd.merge(df_dji['date'], df_dji['close'].div(fst_day).mul(1000000),
                how='outer', left_index=True, right_index=True).set_index('date')
    
    return dji


def ensure_utc_index(df):
    df.index = pd.to_datetime(df.index, utc=True)
    
    return df


def merge_results(
    df_account_value_a2c_agent1, 
    df_account_value_a2c_agent2, 
    df_account_value_sac_agent1,
    df_account_value_sac_agent2,
    MVO_result, 
    dji
    ):
    
    df_result_a2c_agent1 = ensure_utc_index(df_account_value_a2c_agent1.set_index(df_account_value_a2c_agent1.columns[0]))
    df_result_a2c_agent1.columns = ['a2c_agent1']

    df_result_a2c_agent2 = ensure_utc_index(df_account_value_a2c_agent2.set_index(df_account_value_a2c_agent2.columns[0]))
    df_result_a2c_agent2.columns = ['a2c_agent2']

    df_result_sac_agent1 = ensure_utc_index(df_account_value_sac_agent1.set_index(df_account_value_sac_agent1.columns[0]))
    df_result_sac_agent1.columns = ['sac_agent1']

    df_result_sac_agent2 = ensure_utc_index(df_account_value_sac_agent2.set_index(df_account_value_sac_agent2.columns[0]))
    df_result_sac_agent2.columns = ['sac_agent2']

    MVO_result.columns = ['mvo']
    dji.columns = ['dji']

    # Join
    result = df_result_a2c_agent1.copy()
    result = result.join(df_result_a2c_agent2, how='outer')
    result = result.join(df_result_sac_agent1, how='outer')
    result = result.join(df_result_sac_agent2, how='outer')
    result = result.join(MVO_result, how='outer')
    result = result.join(dji, how='outer')

    result = result.fillna(method='bfill')
    col_name = ['A2C Agent 1', 'A2C Agent 2', 'SAC Agent 1', 'SAC Agent 2', 'Mean Var', 'djia']
    result.columns = col_name
    result = result.dropna(subset=['djia'])
    
    return result