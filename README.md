# Stock Analysis Chatbot

## Overview
The Stock Analysis Chatbot is an interactive web application designed to provide real-time stock analysis and insights using advanced technical indicators. Built with Python and Streamlit, this chatbot utilizes natural language processing and data visualization techniques to answer user queries related to stock prices, trends, and technical analysis.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Features
- Comprehensive technical analysis using indicators like **RSI**, **Bollinger Bands**, and **Moving Averages**.
- Interactive chat interface for real-time user queries about stock prices and trends.
- Dynamic visualizations with **Plotly**, including candlestick charts and technical indicators.
- Scalable data pipeline powered by the **yfinance API** for historical and real-time stock data processing.

## Technologies Used
- **Python**: Main programming language for backend logic and data processing.
- **Streamlit**: Framework for creating the web application interface.
- **spaCy NLP**: For natural language processing and query understanding.
- **Plotly**: For interactive data visualization.
- **yfinance API**: For fetching real-time and historical stock market data.
- **Pandas**: For data manipulation and analysis.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/anithag09/StockAnalysisChatbot.git
   ```
   
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the Required Packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the Stock Analysis Chatbot, use the following command:
  
    streamlit run Chatbot.py
 
Once the application is running, open your web browser and navigate to http://localhost:8501 to interact with the chatbot.

## Interacting with the Chatbot
- Ask about the current stock price, trends, or any financial advice.
- You can input stock symbols to fetch specific stock data.
- Explore the visualizations provided for a detailed analysis.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.
