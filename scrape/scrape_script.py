import csv
import os
import json
import hashlib
import time
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import praw
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import tldextract
from config import GROK3_ANONUSERID, GROK3_CHALLENGE, GROK3_SIGNATURE, GROK3_SSO, GROK3_SSO_RW

# Import GrokClient
# Make sure you have installed it: pip install . in the grok3-api directory
try:
    from grok_client import GrokClient
except ImportError:
    print("Warning: grok_client not found. Grok3 scraping will be skipped.")
    GrokClient = None # Set to None if not available

# --------------------------------------------------------------------------------------
REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT"),
    "username": os.getenv("REDDIT_USERNAME"),
    "password": os.getenv("REDDIT_PASSWORD")
}
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket", "finance"]
QUERIES = ["nvidia", "NVDA"]
REDDIT_SCORE_MIN = 50
REDDIT_LOOKBACK_DAYS = 2

RSS_URLS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    "https://feeds-api.dotdashmeredith.com/v1/rss/google/f8466ec3-5044-46bc-94b7-2df65f770eff"
]
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWS_DOMAINS = 'fool.com,etfdailynews.com,marketplace.org,forbes.com,denverpost.com,bostonherald.com,globenewswire.com,ndtv.com,nbcnews.com'

today = datetime.now()
two_days_ago = today - timedelta(days=2)
FROM_DATE = two_days_ago.strftime('%Y-%m-%dT00:00:00')
TO_DATE = today.strftime('%Y-%m-%dT23:59:59')

COMBINED_OUTPUT_FILE = "news.csv"
SEEN_HASH_FILE = "seen_hashes.json"

# --------------------------------------------------------------------------------------

def compute_hash(title, link):
    return hashlib.md5(f"{title}{link}".encode('utf-8')).hexdigest()

def load_seen_hashes():
    if os.path.exists(SEEN_HASH_FILE):
        with open(SEEN_HASH_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_seen_hashes(hashes):
    with open(SEEN_HASH_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(hashes), f, indent=2)

def contains_relevant_keywords(title, content):
    relevant_keywords = ['nvidia', 'nvda', 'stock']
    title = title.lower()
    content = content.lower()
    return any(keyword in title or keyword in content for keyword in relevant_keywords)

# --------------------------------------------------------------------------------------

def extract_specific_source(link):
    if pd.isna(link):
        return 'Unknown'
    link = str(link)
    domain_info = tldextract.extract(link)
    full_domain = domain_info.registered_domain.lower()

    if 'reddit.com' in full_domain:
        match = re.search(r'reddit\.com/r/([^/]+)/', link)
        return match.group(1) if match else 'Reddit'
    if 'yahoo.com' in full_domain:
        return 'Yahoo Finance'
    return domain_info.domain.capitalize() if domain_info.domain else 'Unknown'

def preprocess_new_entries(df):
    df['Full Text'] = df['Full Text'].apply(lambda x: str(x).replace('\n', ' ') if pd.notna(x) else '')
    df['SpecificSource'] = df['Link'].apply(extract_specific_source)
    return df

# --------------------------------------------------------------------------------------

def scrape_reddit():
    reddit = praw.Reddit(**REDDIT_CONFIG)
    min_time = datetime.utcnow() - timedelta(days=REDDIT_LOOKBACK_DAYS)
    min_timestamp = min_time.timestamp()
    results = []

    for sub in SUBREDDITS:
        for query in QUERIES:
            for post in reddit.subreddit(sub).search(query, sort="new"):
                if post.created_utc < min_timestamp or post.score < REDDIT_SCORE_MIN:
                    continue
                if not post.selftext.strip():
                    continue

                dt = datetime.utcfromtimestamp(post.created_utc)
                results.append({
                    "Date and Timestamp": dt.strftime('%Y-%m-%d %H:%M:%S'),
                    "Title": post.title,
                    "Full Text": post.selftext.strip(),
                    "Source": "Reddit",
                    "Link": f"https://reddit.com{post.permalink}"
                })
    print("Reddit done")
    return results

def scrape_rss():
    results = []

    def get_full_content(url, target_class):
        time.sleep(2)
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            soup = BeautifulSoup(r.content, 'html.parser')
            article = soup.find('div', class_=target_class)
            if article:
                return '\n'.join(p.get_text(strip=True) for p in article.find_all('p')) or article.get_text(strip=True)
            return "Article body not found"
        except Exception as e:
            return f"Error: {str(e)}"

    for feed_url in RSS_URLS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            link = entry.get('link')
            title = entry.get('title', 'N/A')
            pub_dt = datetime.fromtimestamp(time.mktime(entry.get('published_parsed'))) \
                     if entry.get('published_parsed') else datetime.utcnow()
            timestamp = pub_dt.strftime('%Y-%m-%d %H:%M:%S')

            class_map = {
                'yahoo.com': 'body yf-1ir6o1g',
                'fool.com': 'article-body',
                'investopedia.com': 'comp article-body mntl-block',
                'nasdaq.com': 'jupiter22-c-article-body',
            }
            target_class = next((v for k, v in class_map.items() if k in link), 'article-body')

            content = get_full_content(link, target_class)

            if contains_relevant_keywords(title, content):
                results.append({
                    "Date and Timestamp": timestamp,
                    "Title": title,
                    "Full Text": content.replace('\n', ' '),
                    "Source": "News",
                    "Link": link
                })
    print("RSS Feed done")
    return results

def scrape_newsapi():
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    results = []

    def fetch_full_article(url):
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            possible_classes = [
                'b6Cr_ article-body-container', 'entry', 'Story_body__ZYOg0 userContent',
                'zox-post-body-wrap', 'sp_txt', 'article-body'
            ]
            for class_name in possible_classes:
                article = soup.find('div', class_=class_name)
                if article:
                    return article.get_text().replace('\n', ' ')
            return "Full article text not available."
        except Exception as e:
            return f"Error: {str(e)}"

    articles = client.get_everything(
        q='Nvidia stock',
        language='en',
        sort_by='publishedAt',
        page_size=80,
        page=1,
        from_param=FROM_DATE,
        to=TO_DATE,
        domains=NEWS_DOMAINS
    )['articles']
    
    for article in articles:
        dt = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        content = fetch_full_article(article['url'])

        if contains_relevant_keywords(article['title'], content):
            results.append({
                "Date and Timestamp": timestamp,
                "Title": article['title'],
                "Full Text": content,
                "Source": "News",
                "Link": article['url']
            })
    print("NewsAPI done")
    return results

def scrape_grok3():
    print("Scraping Grok3 (X) data for NVIDIA...")
    results = []

    if not GrokClient:
        print("GrokClient not available. Skipping Grok3 scraping.")
        return results

    # Check if all necessary cookies are set
    cookies = {
        "x-anonuserid": GROK3_ANONUSERID,
        "x-challenge": GROK3_CHALLENGE,
        "x-signature": GROK3_SIGNATURE,
        "sso": GROK3_SSO,
        "sso-rw": GROK3_SSO_RW
    }

    if not all(cookies.values()):
        print("Warning: Grok3 API cookies are not fully set in environment variables. Skipping Grok3 scraping.")
        return results

    try:
        client = GrokClient(cookies)

        # Craft a precise prompt to minimize hallucination and get structured data
        prompt = f"""
        As a highly analytical financial expert, summarize the key discussions and sentiment among X (Twitter) users regarding NVIDIA stock (NVDA) from the last 24-48 hours. Focus solely on financial implications, stock performance, market sentiment, and any significant news or rumors directly impacting NVDA.

        Structure your response as follows:
        Headline: A concise, catchy headline summarizing the overall sentiment or most significant news.
        Summary Points:
        - Point 1: Briefly describe a key topic or development.
        - Point 2: Briefly describe another key topic or development.
        - ... (add more points as necessary)
        Overall Sentiment: Briefly state the prevailing sentiment (e.g., "Bullish," "Bearish," "Mixed," "Cautious optimism").

        Ensure all information is factual and verifiable based on X discussions. Do NOT invent information or speculate beyond what is being discussed. Prioritize data-driven insights.
        """

        response_text = client.send_message(prompt)
        print(f"Raw Grok3 Response:\n{response_text[:500]}...") # Print first 500 chars

        # --- Parse Grok3's response ---
        # This parsing is critical to get the data into the desired format
        title = "Grok3 X (Twitter) Summary on NVIDIA Stock"
        full_text_content = ""
        overall_sentiment = "Mixed" # Default sentiment

        # Attempt to extract Headline
        headline_match = re.search(r"Headline:\s*(.*)", response_text, re.IGNORECASE)
        if headline_match:
            title = headline_match.group(1).strip()
            # If the headline is just "Grok3 X (Twitter) Summary...", make it more specific
            if "Grok3 X (Twitter) Summary" in title and "NVIDIA" in response_text:
                title = f"Grok3 X Summary: {title}" if len(title) < 50 else title


        # Attempt to extract Summary Points
        summary_points_match = re.search(r"Summary Points:\s*(.*?)(?=\nOverall Sentiment:|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        if summary_points_match:
            summary_points_raw = summary_points_match.group(1).strip()
            # Replace bullet points with numbered points or just concatenate with space
            full_text_content = re.sub(r"-\s*", "", summary_points_raw).replace("\n", " ").strip()
            if full_text_content:
                full_text_content = "X (Twitter) discussions reveal: " + full_text_content
        
        # Attempt to extract Overall Sentiment
        sentiment_match = re.search(r"Overall Sentiment:\s*(.*)", response_text, re.IGNORECASE)
        if sentiment_match:
            overall_sentiment = sentiment_match.group(1).strip()
            if overall_sentiment: # Append sentiment to full_text_content if found
                full_text_content += f" Overall sentiment on X: {overall_sentiment}."
        
        # Fallback if parsing was poor
        if not full_text_content and response_text:
            full_text_content = "Grok3's summary from X (Twitter) discussions about NVIDIA: " + response_text.replace('\n', ' ').strip()
            title = f"Grok3 X Summary on NVIDIA Stock: {overall_sentiment}"

        # Ensure title and content are not empty
        if not title:
            title = "Grok3 X (Twitter) Summary on NVIDIA Stock"
        if not full_text_content:
            full_text_content = "Could not extract structured content from Grok3's response. Raw response: " + response_text.replace('\n', ' ').strip()

        # Add a unique "link" for Grok3 entries, as they don't have a direct URL
        # We can use a hash of the content to make it unique if date is the same
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        grok_link_hash = hashlib.md5(f"{current_timestamp}{title}{full_text_content}".encode('utf-8')).hexdigest()
        grok_link = f"[https://grok.com/summary/](https://grok.com/summary/){grok_link_hash}"


        if contains_relevant_keywords(title, full_text_content):
            results.append({
                "Date and Timestamp": current_timestamp,
                "Title": title,
                "Full Text": full_text_content,
                "Source": "Grok3", # New source type
                "Link": grok_link
            })
    except Exception as e:
        print(f"Error scraping Grok3: {e}")
        # Optionally, log the full error or response_text for debugging
        # print(f"Failed Grok3 Response: {response_text}")
    
    print("Grok3 scraping done.")
    return results
# --------------------------------------------------------------------------------------

def main():
    all_data = []
    seen_hashes = load_seen_hashes()

    for scraper in [scrape_reddit, scrape_newsapi, scrape_rss, scrape_grok3]:
        data = scraper()
        for item in data:
            item_hash = compute_hash(item['Title'], item['Link'])
            if item_hash not in seen_hashes:
                all_data.append(item)
                seen_hashes.add(item_hash)

    if all_data:
        new_data_df = pd.DataFrame(all_data)
        new_data_df = preprocess_new_entries(new_data_df)
        COLUMN_ORDER = ['Date and Timestamp', 'Title', 'Full Text', 'Source', 'SpecificSource', 'Link']
        new_data_df = new_data_df[COLUMN_ORDER]

        new_data_df['Date and Timestamp'] = pd.to_datetime(new_data_df['Date and Timestamp'])
        new_data_df = new_data_df.sort_values(by='Date and Timestamp', ascending=True)
        new_data_df = new_data_df.drop_duplicates(subset='Title', keep='first')

        if os.path.exists(COMBINED_OUTPUT_FILE):
            new_data_df.to_csv(COMBINED_OUTPUT_FILE, mode='a', header=False, index=False, sep=';', quoting=csv.QUOTE_ALL)
        else:
            new_data_df.to_csv(COMBINED_OUTPUT_FILE, index=False, sep=';', quoting=csv.QUOTE_ALL)

        save_seen_hashes(seen_hashes)
    else:
        print("No new data scraped.")

if __name__ == "__main__":
    main()
