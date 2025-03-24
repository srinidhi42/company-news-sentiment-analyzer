import requests
import streamlit as st

# Set up your NewsAPI key (ensure you have a .streamlit/secrets.toml or use fallback)
try:
    NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "3f07fbdbdfae4a52bfc84f58f34fd57c")
except Exception:
    NEWS_API_KEY = "3f07fbdbdfae4a52bfc84f58f34fd57c"

def fetch_news(company):
    """
    Fetches news articles from NewsAPI for a given company.
    Returns a list of dictionaries with keys "Title", "Link", and "Summary".
    """
    url = (
        f"https://newsapi.org/v2/everything?q={company}"
        f"&pageSize=10&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    )
    
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error retrieving news articles. Please try again later.")
        return []
    
    data = response.json()
    if data.get("status") != "ok":
        st.error("Error retrieving news articles. Please try again later.")
        return []
    
    articles = []
    for article in data.get("articles", []):
        articles.append({
            "Title": article.get("title"),
            "Link": article.get("url"),
            "Summary": article.get("description")
        })
    return articles
