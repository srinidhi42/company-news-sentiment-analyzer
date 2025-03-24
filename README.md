# company-news-sentiment-analyzer

The Company News Sentiment Analyzer is a Streamlit web application that fetches news articles about a given company, performs advanced sentiment and topic analysis using state-of-the-art NLP models, and generates a comparative analysis report. The project also features a text-to-speech (TTS) functionality that converts the analysis into Hindi audio.

This project is structured into three main components:

app.py: The main application file handling the user interface and orchestration of analysis tasks.

utils.py: Contains utility functions for sentiment analysis, topic extraction, coverage comparison, and TTS conversion.

api.py: Handles API requests (using NewsAPI) to fetch news articles based on the company name.

## Installation & Setup
Prerequisites
Python 3.8 or later

Git

Steps
## Clone the Repository

Open your terminal and run:
git clone https://github.com/srinidhi42/company-news-sentiment-analyzer.git
cd company-news-sentiment-analyzer

## Create a Virtual Environment

bash

python -m venv venv
# Activate the virtual environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

## Install Dependencies

Make sure you have a requirements.txt file containing:
gTTS
deep-translator
requests
streamlit
torch
transformers

pip install -r requirements.txt

## Setup instructions
Running the Application

To start the application, execute:

bash

streamlit run app.py

Using the App

Search: Enter a company name (e.g., "Apple") in the text input and click Search. The app will fetch the latest news articles related to that company.

Analysis: The application performs sentiment and topic analysis, displays a comparative report in JSON format, and shows details of the extracted articles.

TTS Feature: Click the Listen in Hindi button to hear the analysis read out in Hindi. The application translates the text and uses TTS to generate an audio file.

Query System: (Bonus) Enter a keyword or phrase in the query section to filter articles for a more detailed report.



