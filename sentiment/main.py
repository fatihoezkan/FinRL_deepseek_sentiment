from config import RAW_DATA_CSV, TEMP_PROCESSED_JSON, NEWS_WITH_SCORE_CSV, TEMP_DATE_RISK_CSV
from data_preprocessing import data_preprocessing
from risk_score_generation import get_all_scores, append_score_to_csv
from risk_score_aggregation import aggregate_risk_score
from risk_score_validation import validate_all_scores, regeneration
import json
import sys

if __name__ == "__main__":
    try:
        # Extract yesterday's news from news.csv and save it as JSON file
        data_preprocessing(path=RAW_DATA_CSV)
        print("\n--- Data preprocessing completed ---\n")

        # Get the processed data JSON file
        with open(TEMP_PROCESSED_JSON, 'r') as f:
            json_data = json.load(f)
        print("\n--- Processed JSON Data loaded successfully ---\n")
        
        # Generate risk score for each news
        scored_articles = get_all_scores(json_data)
        print("\n--- Risk Score Generation Completed ---\n")

        # Validate risk scores
        validated_articles = validate_all_scores(scored_articles)
        print("\n--- Validation Completed ---\n")

        # Regeneration
        regenerated_articles = regeneration(validated_articles)
        print("\n--- Regeneration Completed ---\n")

        # Extract risk scores from all articles
        risk_scores = [article['risk_score'] for article in regenerated_articles]
        
        # Append new data with risk score for tracing purposes
        append_score_to_csv(json_data, risk_scores, NEWS_WITH_SCORE_CSV)
        print("\n--- Appending risk score to CSV Completed ---\n")
        
        # Aggregate risk score
        aggregate_risk_score(TEMP_DATE_RISK_CSV)
        print("\n--- Aggregating risk score Completed ---\n")
    except Exception as e:
        print(f"Error occured in main.py -> {e}")
        sys.exit(1)