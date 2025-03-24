import streamlit as st
import json
from collections import Counter
from api import fetch_news
from utils import (
    analyze_sentiment_advanced,
    generate_coverage_differences_all,
    extract_topics_from_summary,
    generate_final_sentiment_analysis,
    generate_hindi_tts
)

def main():
    st.title("Company News Sentiment Analyzer")
    st.markdown("""
    Enter a company name to fetch news articles related to it, analyze their sentiment,
    and receive a structured comparative sentiment report across multiple articles.
    """)

    company_name = st.text_input("Enter the Company Name:")

    if st.button("Search"):
        if company_name.strip() == "":
            st.error("Please enter a valid company name.")
        else:
            st.info(f"Searching for news articles about '{company_name}'...")
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(company_name)
            if articles:
                # Store the articles in session state for later use
                st.session_state["articles"] = articles

                sentiment_distribution = {"POSITIVE": 0, "NEGATIVE": 0, "Neutral": 0}
                all_summaries = []
                candidate_topics = [
                    "Electric Vehicles", "Stock Market", "Innovation", "Regulations",
                    "Autonomous Vehicles", "Financial Performance", "Legal Issues",
                    "Technology", "Market Growth", "Environmental Impact"
                ]
                all_topics = []  # List to store topics from all articles

                # Process each article
                for article in articles:
                    text = article.get("Summary") or article.get("Title")
                    sentiment = analyze_sentiment_advanced(text)
                    article["Sentiment"] = sentiment
                    sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
                    all_summaries.append(text)
                    topics = extract_topics_from_summary(text, candidate_topics)
                    article["Topics"] = topics
                    all_topics.extend(topics)
                
                st.markdown("#### Extracted Articles")
                st.json({
                    "Company": company_name,
                    "Articles": articles
                })

                # Comparative Analysis
                coverage_diff = generate_coverage_differences_all(all_summaries)
                topic_counts = Counter(all_topics)
                common_topics = [topic for topic, count in topic_counts.items() if count >= len(articles)/2]
                
                positive_topics = []
                negative_topics = []
                for article in articles:
                    if article["Sentiment"] == "POSITIVE":
                        positive_topics.extend(article.get("Topics", []))
                    elif article["Sentiment"] == "NEGATIVE":
                        negative_topics.extend(article.get("Topics", []))
                unique_positive = list(set(positive_topics) - set(negative_topics) - set(common_topics))
                unique_negative = list(set(negative_topics) - set(positive_topics) - set(common_topics))
                
                comparative_sentiment_score = {
                    "Sentiment Distribution": sentiment_distribution,
                    "Coverage Differences": coverage_diff,
                    "Topic Overlap": {
                        "Common Topics": common_topics,
                        "Unique Topics in Positive Coverage": unique_positive,
                        "Unique Topics in Negative Coverage": unique_negative
                    }
                }
                final_analysis = generate_final_sentiment_analysis(sentiment_distribution)
                comparative_output = {
                    "Comparative Sentiment Score": comparative_sentiment_score,
                    "Final Sentiment Analysis": final_analysis
                }
                
                st.markdown("### Comparative Analysis (Across all 10 articles)")
                st.json(comparative_output)
                st.success("News articles successfully extracted and analyzed!")
                
                # Save the comparative output in session state for TTS and queries
                st.session_state["comparative_output"] = comparative_output

    # TTS Button for Hindi audio (using stored analysis)
    if st.button("Listen in Hindi"):
        if "comparative_output" in st.session_state:
            comparative_text = json.dumps(st.session_state["comparative_output"], indent=4)
            # Remove unwanted symbols for TTS clarity
            comparative_text = comparative_text.replace("{", "").replace("}", "").replace(">", "").replace("-", "").replace(":", "")
            audio_file = generate_hindi_tts(comparative_text)
            st.audio(audio_file, format="audio/mp3")
        else:
            st.error("No analysis available. Please run a search first.")
    
    # --- BONUS: Detailed Analysis Reporting & Querying System ---
    if "articles" in st.session_state:
        st.markdown("### Query Your News Stories")
        query = st.text_input("Enter a keyword or phrase to filter the articles:")
        if query:
            filtered_articles = [
                article for article in st.session_state["articles"]
                if query.lower() in (article.get("Title", "").lower() + " " +
                                     (article.get("Summary") or "").lower() + " " +
                                     " ".join(article.get("Topics", [])).lower())
            ]
            if filtered_articles:
                st.markdown("#### Filtered Articles")
                st.json(filtered_articles)
            else:
                st.warning("No articles match your query.")

if __name__ == "__main__":
    main()
