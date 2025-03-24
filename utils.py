from transformers import pipeline

from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator

# Initialize pipelines globally
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# For coverage differences, use a generation model (here t5-small)
generation_pipeline = pipeline("text2text-generation", model="t5-small", max_length=300)
# For topic extraction, using zero-shot classification
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_sentiment_advanced(text):
    """
    Uses a transformer-based sentiment analysis pipeline.
    Returns the sentiment label ("POSITIVE", "NEGATIVE", or "Neutral").
    """
    if not text:
        return "Neutral"
    result = sentiment_pipeline(text)
    return result[0]["label"]

def generate_coverage_differences_all(summaries):
    """
    Uses a text2text-generation model to generate a concise one-line coverage comparison and impact statement.
    The prompt instructs the model to provide a one-line comparison summary of how news coverage differs overall,
    followed by a brief one-line impact statement on the overall public perception.
    
    Returns a dictionary with "Comparison" and "Impact" keys.
    """
    # Aggregate summaries and limit the length
    combined = " || ".join(summaries)
    combined = combined[:1000]  # Limit to first 1000 characters
    prompt = (
        f"Based on the following news article summaries, provide a concise one-line comparison summarizing how the news coverage differs overall, "
        f"followed by a brief one-line impact statement on the overall public perception.\n\n"
        f"Format your answer as: 'Comparison: <comparison text> Impact: <impact text>'\n"
        f"Summaries: {combined}"
    )
    output = generation_pipeline(prompt, do_sample=False)
    generated = output[0]["generated_text"].strip()
    
    # First, try splitting by the explicit keyword "Impact:"
    if "Impact:" in generated:
        parts = generated.split("Impact:")
        comparison_text = parts[0].replace("Comparison:", "").strip()
        impact_text = parts[1].strip()
    # If not, check if the output uses "||" as a separator
    elif "||" in generated:
        parts = generated.split("||")
        if len(parts) >= 2:
            comparison_text = parts[0].strip()
            impact_text = parts[1].strip()
        else:
            comparison_text = generated.strip()
            impact_text = ""
    else:
        comparison_text = generated.strip()
        impact_text = ""
    
    return {"Comparison": comparison_text, "Impact": impact_text}



def extract_topics_from_summary(summary, candidate_topics, threshold=None):
    """
    Uses zero-shot classification to identify which candidate topics appear in the summary.
    
    Args:
        summary (str): The article summary to analyze
        candidate_topics (list): List of potential topics to detect
        threshold (float, optional): Confidence threshold. If None, uses dynamic thresholding
        
    Returns:
        dict: Contains topics with scores and metadata about the classification
    """
    if not summary.strip():
        return {"topics": [], "confidence": 0, "detection_method": "empty_input"}
    
    # Group similar topics to improve classification accuracy
    topic_groups = {}
    for topic in candidate_topics:
        # Create simpler key for topic groups (e.g., "Market Growth" and "Growth" might be related)
        key_words = set(word.lower() for word in topic.split() if len(word) > 3)
        for existing_group in topic_groups:
            existing_words = set(word.lower() for word in existing_group.split() if len(word) > 3)
            # If there's word overlap, group them
            if key_words.intersection(existing_words):
                if existing_group in topic_groups:
                    topic_groups[existing_group].append(topic)
                break
        else:
            topic_groups[topic] = [topic]
    
    # Flatten grouped topics for classification
    grouped_topics = list(topic_groups.keys())
    
    # First pass: classify with grouped topics
    result = zero_shot_pipeline(summary, candidate_labels=grouped_topics)
    
    # Set threshold dynamically if not specified
    if threshold is None:
        # Use different strategies based on the distribution of scores
        scores = result["scores"]
        if len(scores) > 0:
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)
            # If there's a clear winner, use a relative threshold
            if max_score > 0.7 and max_score > 1.5 * mean_score:
                threshold = max_score * 0.7  # 70% of the max score
            else:
                # Otherwise use a fixed but reasonable threshold
                threshold = 0.5
        else:
            threshold = 0.5
    
    # Get topics above threshold
    selected_topics = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= threshold:
            # Expand grouped topics if necessary
            if label in topic_groups and len(topic_groups[label]) > 1:
                # Second pass: refine within the group
                group_result = zero_shot_pipeline(summary, candidate_labels=topic_groups[label])
                for group_label, group_score in zip(group_result["labels"], group_result["scores"]):
                    if group_score >= threshold:
                        selected_topics.append({"topic": group_label, "score": float(group_score)})
            else:
                selected_topics.append({"topic": label, "score": float(score)})
    
    # Sort by score for better presentation
    selected_topics.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "topics": [t["topic"] for t in selected_topics],
        "detailed": selected_topics,
        "threshold_used": threshold,
        "topic_count": len(selected_topics)
    }

def generate_final_sentiment_analysis(sentiment_counts, article_metadata=None):
    """
    Generates a comprehensive sentiment analysis based on the sentiment distribution
    and optional metadata about the articles.
    
    Args:
        sentiment_counts (dict): Counts of different sentiment categories
        article_metadata (list, optional): Optional list of metadata for each article 
                                           (e.g., source, date, title)
    
    Returns:
        dict: Contains overall analysis and detailed breakdowns
    """
    positive = sentiment_counts.get("POSITIVE", 0)
    negative = sentiment_counts.get("NEGATIVE", 0)
    neutral = sentiment_counts.get("Neutral", 0)
    total = positive + negative + neutral
    
    if total == 0:
        return {
            "overall": "No sentiment data available for analysis.",
            "distribution": {},
            "confidence": 0
        }
    
    # Calculate percentages for better comparison
    sentiment_percentages = {
        "POSITIVE": round((positive / total) * 100, 1),
        "NEGATIVE": round((negative / total) * 100, 1),
        "NEUTRAL": round((neutral / total) * 100, 1)
    }
    
    # Calculate the strength of the sentiment bias
    # (how far from an even distribution)
    max_percentage = max(sentiment_percentages.values())
    even_distribution = 100 / len(sentiment_percentages)
    bias_strength = round((max_percentage - even_distribution) / even_distribution, 2)
    
    # Generate more nuanced analysis with confidence levels
    if bias_strength < 0.2:
        confidence = "low"
        overall = "The news coverage appears balanced, with no strong sentiment bias."
    elif 0.2 <= bias_strength < 0.5:
        confidence = "moderate"
        if sentiment_percentages["POSITIVE"] > sentiment_percentages["NEGATIVE"]:
            overall = "The coverage leans somewhat positive, suggesting cautious optimism."
        elif sentiment_percentages["NEGATIVE"] > sentiment_percentages["POSITIVE"]:
            overall = "The coverage tends slightly negative, indicating some concerns."
        else:
            overall = "The coverage is notably neutral, focusing on factual reporting."
    else:
        confidence = "high"
        if sentiment_percentages["POSITIVE"] > sentiment_percentages["NEGATIVE"]:
            overall = "The overall news coverage is predominantly positive, suggesting strong market confidence."
        elif sentiment_percentages["NEGATIVE"] > sentiment_percentages["POSITIVE"]:
            overall = "The coverage is distinctly negative, reflecting significant market concerns."
        else:
            overall = "The coverage is remarkably neutral, suggesting deliberate objective reporting."
    
    # Add temporal analysis if metadata is available
    trend_analysis = None
    if article_metadata and len(article_metadata) > 1:
        # Sort by date if available
        try:
            sorted_data = sorted(
                [(meta.get("date", ""), meta.get("sentiment", "Neutral")) 
                 for meta in article_metadata if "date" in meta],
                key=lambda x: x[0]
            )
            
            if len(sorted_data) >= 3:  # Need at least 3 points for a trend
                # Simple trend analysis
                early_sentiments = [s for _, s in sorted_data[:len(sorted_data)//2]]
                late_sentiments = [s for _, s in sorted_data[len(sorted_data)//2:]]
                
                early_positive = early_sentiments.count("POSITIVE")
                early_negative = early_sentiments.count("NEGATIVE")
                late_positive = late_sentiments.count("POSITIVE")
                late_negative = late_sentiments.count("NEGATIVE")
                
                if early_positive < late_positive and early_negative > late_negative:
                    trend_analysis = "The sentiment shows improvement over time, with more recent coverage being more positive."
                elif early_positive > late_positive and early_negative < late_negative:
                    trend_analysis = "The sentiment has deteriorated over time, with more recent coverage being more negative."
                else:
                    trend_analysis = "The sentiment has remained relatively consistent over time."
        except:
            # Skip temporal analysis if there's any issue
            pass
    
    result = {
        "overall": overall,
        "distribution": sentiment_percentages,
        "confidence": confidence,
        "bias_strength": bias_strength,
        "dominant_sentiment": max(sentiment_percentages, key=sentiment_percentages.get)
    }
    
    if trend_analysis:
        result["trend_analysis"] = trend_analysis
        
    return result
    
    
def generate_hindi_tts(text):
    """
    Translates the input text to Hindi and then converts it to speech using gTTS.
    Returns an in-memory audio file (BytesIO) that can be played in Streamlit.
    """
    # Replace the translator code
    translated_text = GoogleTranslator(source='auto', target='hi').translate(text)
    
    tts = gTTS(translated_text, lang="hi")
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp
