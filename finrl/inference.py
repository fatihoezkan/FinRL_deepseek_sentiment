from utils import *
from config import RESULTS_CSV
import sys


def get_inference():
    """
    This function executes the steps in the inference stage.
    It loads the train_data.csv and trade_data.csv obtained from the processing stage, and loads the aggregated_risk_scores.csv from the /sentiment part.
    It runs the model prediction for Agent 1 (only stock data) and Agent 2 (stock data + sentiment data).
    It calculates the MVO and loads the DJIA benchmark index hourly data as a base comparison.
    It merges the results from Agent 1, Agent 2, MVO, and DJIA into results.csv to be plotted in the final dashboard.

    Args:
        None
        
    """ 
    try:
        # Load train and trade data
        trade = load_trade()
        print("Loaded train and trade csv.")
        
        # Load the pre-trained A2C model
        trained_a2c = load_trained_a2c()
        print("Loaded trained A2C model.")
        
        # Load the pre-trained SAC model
        trained_sac = load_trained_sac()
        print("Loaded trained SAC model.")

        #load the pre-trained PPO model
        trained_ppo = load_trained_ppo()
        print("Loaded trained PPO model.")

        # Load the pre-trained TD3 model
        trained_td3 = load_trained_td3()
        print("Loaded trained TD3 model.")
         
        # Load aggregated_risk_scores
        trade_sentiment = load_aggregated_risk_score(trade)
        print("Loaded aggregated risk scores and merged with trade data.")

        # Load trade turbulence data
        trade_turbulence = load_sentiment_and_turbulence(trade)
        print("Loaded trade turbulence data.")

        # Predict A2C Agent 1
        df_account_value_a2c_agent1, _ = predict_agent_1(trade, trained_a2c)
        print("A2C Agent 1 prediction done.")
        
        # Predict SAC Agent 1
        df_account_value_sac_agent1, _ = predict_agent_1(trade, trained_sac)
        print("SAC Agent 1 prediction done.")
        
        # Predict PPO Agent 1
        df_account_value_ppo_agent1, _ = predict_agent_1(trade, trained_ppo)
        print("PPO Agent 1 prediction done.")

        # Predict TD3 Agent 1
        df_account_value_td3_agent1, _ = predict_agent_1(trade, trained_td3)
        print("TD3 Agent 1 prediction done.")

        # Predict A2C Agent 2
        df_account_value_a2c_agent2, _ = predict_agent_2(trade, trained_a2c, trade_sentiment)
        print("A2C Agent 2 prediction done.")
        
        # Predict SAC Agent 2
        df_account_value_sac_agent2, _ = predict_agent_2(trade, trained_sac, trade_sentiment)
        print("SAC Agent 2 prediction done.")

        # Predict PPO Agent 2
        df_account_value_ppo_agent2, _ = predict_agent_2(trade, trained_ppo, trade_sentiment)

        print("PPO Agent 2 prediction done.")
        # Predict TD3 Agent 2
        df_account_value_td3_agent2, _ = predict_agent_2(trade, trained_td3, trade_sentiment)

        #predict A2C Agent 3
        df_account_value_a2c_agent3, _ = predict_agent_3(trade, trained_a2c, trade_turbulence)
        print("A2C Agent 3 prediction done.")

        # Predict SAC Agent 3
        df_account_value_sac_agent3, _ = predict_agent_3(trade, trained_sac, trade_turbulence)
        print("SAC Agent 3 prediction done.")

        # Predict PPO Agent 3
        df_account_value_ppo_agent3, _ = predict_agent_3(trade, trained_ppo, trade_turbulence)
        print("PPO Agent 3 prediction done.")

        # Predict TD3 Agent 3
        df_account_value_td3_agent3, _ = predict_agent_3(trade, trained_td3, trade_turbulence)

        # Calculate Mean Variance Optimization (MVO)
        StockData, arStockPrices, rows, cols = calculate_mvo(trade)
        
        # Calculate Mean Returns and Covariance Matrix
        meanReturns, covReturns = calculate_mean_cov(arStockPrices, rows, cols)
        print("Mean Returns:", meanReturns)
        print("Covariance Matrix:", covReturns)
        
        # Calculate Efficient Frontier
        MVO_result = calculate_efficient_frontier(meanReturns, covReturns, trade, StockData)
        print("MVO calculation done.")
        
        # Get hourly data from DJIA benchmark index
        dji = get_djia_index(trade)
        print("Loaded DJIA hourly data.")
        
        # Merge results
        result = merge_results(
                    df_account_value_a2c_agent1, 
                    df_account_value_a2c_agent2,
                    df_account_value_a2c_agent3, 
                    df_account_value_sac_agent1,
                    df_account_value_sac_agent2,
                    df_account_value_sac_agent3,
                    df_account_value_ppo_agent1,
                    df_account_value_ppo_agent2,
                    df_account_value_ppo_agent3,
                    df_account_value_td3_agent1,
                    df_account_value_td3_agent2,
                    df_account_value_td3_agent3,
                    MVO_result, 
                    dji
                )
        
        # Save results to csv
        result.to_csv(RESULTS_CSV)
        print("Results merged and saved as results.csv")
        
        
    except Exception as e:
        print(f"[Error in finrl inference.py] -> {e}")
        sys.exit(1)