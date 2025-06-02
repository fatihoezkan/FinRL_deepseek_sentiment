# scrape/grok_scrapper.py
import os
import re
import hashlib
from datetime import datetime, timedelta
import pandas as pd # For pd.isna in extract_specific_source, though not strictly needed if link is guaranteed non-None from Grok
import tldextract
import csv
import json

# Import GrokClient from the installed package
try:
    from grok_client import GrokClient
except ImportError:
    print("Error: 'grok_client' package not found. Please ensure it's installed correctly.")
    print("Refer to the grok3-api README for installation instructions: https://github.com/mem0ai/grok3-api")
    GrokClient = None # Set to None if not available

try:
    from config import GROK3_ANONUSERID, GROK3_CHALLENGE, GROK3_SIGNATURE, GROK3_SSO, GROK3_SSO_RW
except ImportError:
    print("Error: Could not import Grok3 API environment variables from 'config.py'.")
    print("Please ensure 'config.py' exists and contains the necessary GROK3_ variables.")
    GROK3_ANONUSERID = None
    GROK3_CHALLENGE = None
    GROK3_SIGNATURE = None
    GROK3_SSO = None
    GROK3_SSO_RW = None

def compute_hash(title, link):
    """Computes a hash for a given title and link to identify unique entries."""
    return hashlib.md5(f"{title}{link}".encode('utf-8')).hexdigest()

def contains_relevant_keywords(title, content):
    """Checks if the title or content contains keywords relevant to NVIDIA."""
    relevant_keywords = ['nvidia', 'nvda', 'stock', 'chip', 'ai', 'gpu']
    title = str(title).lower()
    content = str(content).lower()
    return any(keyword in title or keyword in content for keyword in relevant_keywords)

def extract_specific_source(link):
    """Extracts the specific source, tailored for Grok3/X data."""
    if isinstance(link, str) and ('grok.com' in link or 'x.com' in link):
        return 'X'
    if pd.isna(link):
        return 'Unknown'
    domain_info = tldextract.extract(link)
    return domain_info.domain.capitalize() if domain_info.domain else 'Unknown'

def scrape_grok3():
    """
    Scrapes data from Grok3 (X) about NVIDIA stock.

    Returns:
        list: A list of dictionaries, each representing a news entry
              formatted as:
              {
                  "Date and Timestamp": "YYYY-MM-DD HH:MM:SS",
                  "Title": "Concise headline from Grok3",
                  "Full Text": "Detailed summary from Grok3",
                  "Source": "Grok3",
                  "SpecificSource": "X",
                  "Link": "Unique Grok3 generated link"
              }
    """
    print("Initiating Grok3 (X) data scraping for NVIDIA...")
    results = []

    if not GrokClient:
        print("GrokClient is not initialized. Cannot proceed with Grok3 scraping.")
        return results

    # Check if all cookies are set
    cookies = {
        "x-anonuserid": GROK3_ANONUSERID,
        "x-challenge": GROK3_CHALLENGE,
        "x-signature": GROK3_SIGNATURE,
        "sso": GROK3_SSO,
        "sso-rw": GROK3_SSO_RW
    }

    print(f"\nCookies:\n{json.dumps(cookies, indent=4)}")

    if not all(cookies.values()):
        print("Warning: Grok3 API cookies are not fully set in environment variables.")
        print("Please ensure GROK3_ANONUSERID, GROK3_CHALLENGE, GROK3_SIGNATURE, GROK3_SSO, GROK3_SSO_RW are configured.")
        return results

    try:
        client = GrokClient(cookies)
        # Testing
        response = client.send_message("Write me a poem")
        print(f"the response is: {response}")

        # Requesting a summary for the last 24-48 hours.
        prompt = f"""
        As a highly analytical financial expert, summarize the key discussions and sentiment among X (Twitter) users regarding NVIDIA stock (NVDA) from the last 24-48 hours. Focus solely on financial implications, stock performance, market sentiment, and any significant news or rumors directly impacting NVDA.

        Structure your response as follows:
        Headline: A concise, catchy headline summarizing the overall sentiment or most significant news.
        Summary Points:
        - Point 1: Briefly describe a key topic or development.
        - Point 2: Briefly describe another key topic or development.
        - ... (add more points as necessary)
        Overall Sentiment: Briefly state the prevailing sentiment (e.g., "Bullish," "Bearish," "Mixed," "Cautious optimism").

        Ensure all information is factual and verifiable based on X discussions. Do NOT invent information or speculate beyond what is being discussed. Prioritize data-driven insights. If there's no significant discussion, state that.
        """
        # response_text = client.send_message(prompt)
        response_text = client.send_message("Give me a summary of Twitter discussions about NVIDIA stock.")
        print(f"Raw Grok3 Response:\n{response_text}\n")

        # --- Parse Grok3's response ---
        title = "Grok3 X (Twitter) Summary on NVIDIA Stock"
        full_text_content = ""
        overall_sentiment = "Mixed" # Default sentiment if not found

        # Extract Headline
        headline_match = re.search(r"Headline:\s*(.*)", response_text, re.IGNORECASE)
        if headline_match:
            title = headline_match.group(1).strip()
            if "Grok3 X (Twitter) Summary" in title and "NVIDIA" in response_text:
                title = f"Grok3 X Summary: {title}" if len(title) < 50 else title
            elif "Grok3 X (Twitter) Summary on NVIDIA Stock" not in title:
                title = f"Grok3 X (Twitter) Summary on NVIDIA Stock: {title}"

        # Extract Summary Points
        summary_points_match = re.search(r"Summary Points:\s*(.*?)(?=\nOverall Sentiment:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        if summary_points_match:
            summary_points_raw = summary_points_match.group(1).strip()
            full_text_content = re.sub(r"[\t\n\r\f\v-]+", " ", summary_points_raw).strip()
            if full_text_content:
                full_text_content = "X (Twitter) discussions reveal: " + full_text_content

        # Extract Overall Sentiment
        sentiment_match = re.search(r"Overall Sentiment:\s*(.*)", response_text, re.IGNORECASE)
        if sentiment_match:
            overall_sentiment = sentiment_match.group(1).strip()
            if overall_sentiment:
                # Add a period if content exists, then append sentiment
                if full_text_content and not full_text_content.endswith('.'):
                    full_text_content += "."
                full_text_content += f" Overall sentiment on X: {overall_sentiment}."
        
        if not full_text_content or (not summary_points_match and not sentiment_match):
            full_text_content = "Grok3's summary from X (Twitter) discussions about NVIDIA: " + response_text.replace('\n', ' ').strip()
            if not title or "Grok3 X (Twitter) Summary on NVIDIA Stock" in title: # Only change title if it's default or generic
                title = f"Grok3 X Summary on NVIDIA Stock ({overall_sentiment})" if overall_sentiment != "Mixed" else "Grok3 X Summary on NVIDIA Stock"
        
        if not title:
            title = "Grok3 X (Twitter) Summary on NVIDIA Stock (No specific headline)"
        if not full_text_content:
            full_text_content = "Grok3 returned no discernible content from X (Twitter) discussions on NVIDIA. Raw response: " + response_text.replace('\n', ' ').strip()


        current_timestamp_dt = datetime.now()
        current_timestamp_str = current_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
        grok_link_hash = hashlib.md5(f"{current_timestamp_str}{title}{full_text_content}".encode('utf-8')).hexdigest()
        grok_link = f"https://grok.com/summary/{current_timestamp_dt.strftime('%Y%m%d%H%M%S')}/{grok_link_hash}"


        # Only add if relevant keywords are present in the generated content
        if contains_relevant_keywords(title, full_text_content):
            results.append({
                "Date and Timestamp": current_timestamp_str,
                "Title": title,
                "Full Text": full_text_content,
                "Source": "Grok3",
                "Link": grok_link
            })
            # Add SpecificSource here as well, so it's part of the raw data dict
            results[-1]["SpecificSource"] = extract_specific_source(grok_link)
        else:
            print("Grok3 response did not contain relevant keywords for NVIDIA. Skipping.")

    except Exception as e:
        print(f"Error during Grok3 scraping: {e}")
        # print(f"Failed Grok3 Raw Response: {response_text if 'response_text' in locals() else 'N/A'}")
    
    print("Grok3 scraping process completed.")
    return results

if __name__ == "__main__":
    print("--- Running Grok3 Scraper in Isolation ---")

    scraped_grok_data = scrape_grok3()

    if scraped_grok_data:
        print("\n--- Scraped Grok3 Data ---")
        for entry in scraped_grok_data:
            print(f"Date: {entry['Date and Timestamp']}")
            print(f"Title: {entry['Title']}")
            print(f"Full Text: {entry['Full Text']}")
            print(f"Source: {entry['Source']}")
            print(f"SpecificSource: {entry['SpecificSource']}")
            print(f"Link: {entry['Link']}")
            print("-" * 30)
        
        COLUMN_ORDER = ['Date and Timestamp', 'Title', 'Full Text', 'Source', 'SpecificSource', 'Link']
        df = pd.DataFrame(scraped_grok_data)
        
        if 'SpecificSource' not in df.columns:
            df['SpecificSource'] = df['Link'].apply(extract_specific_source)

        df = df[COLUMN_ORDER]
        temp_grok_output_file = "temp_grok_news.csv"
        df.to_csv(temp_grok_output_file, index=False, sep=';', quoting=csv.QUOTE_ALL)
        print(f"\nScraped Grok3 data saved to {temp_grok_output_file}")
    else:
        print("No data was scraped from Grok3.")
