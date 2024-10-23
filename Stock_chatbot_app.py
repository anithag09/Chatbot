import streamlit as st
import pandas as pd
import numpy as np
import spacy
import random
import base64
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# Set page config for wide layout and a title
st.set_page_config(page_title="Stock Analysis", layout="wide")

# Load spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Load the CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the function to apply CSS
local_css("style.css")


# Define intents for the chatbot
intents = {
    "greet": ["hello", "hi", "hey", "greetings"],
    "goodbye": ["bye", "goodbye", "see you", "farewell"],
    "ask_financial_advice": ["need financial advice", "help with finances", "investment advice", "money management"],
    "ask_stock_price": ["stock price", "share value", "stock market", "current price", "price now"],
    "ask_stock_trend": ["trend", "movement", "stock performance", "price trend", "analysis"],
    "ask_stock_volume": ["volume", "trading volume", "stock volume"],
}

# Responses based on intents
responses = {
    "greet": [
        "Hello! How can I assist you with stock information today?",
        "Hi there! What would you like to know about stock data?"
    ],
    "goodbye": [
        "Goodbye! Feel free to return for more stock insights.",
        "Take care! Keep tracking those market movements!"
    ],
    "ask_financial_advice": [
        "Based on recent trends, Reliance stock has shown significant movement. However, for personalized investment advice, please consult a financial advisor."
    ],
    "default": [
        "I'm not sure I understand. You can ask about stock price, trends, or trading volume."
    ]
}

# Helper function to get the intent from user input
def get_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intents.items():
        if any(keyword in user_input for keyword in keywords):
            return intent
    return "default"

# Chatbot function for generating responses based on the intent
def generate_stock_response(stock_handler, intent):
    if intent == "ask_stock_price":
        current_price = stock_handler.get_current_price()
        price_change = stock_handler.get_price_change()
        return f"The current stock price is ₹{current_price:.2f} ({price_change:+.2f}% change)."
    
    elif intent == "ask_stock_trend":
        analysis = stock_handler.get_technical_analysis()
        return f"""Technical Analysis Summary:
        - RSI: {analysis['rsi']['value']:.1f} ({analysis['rsi']['signal']})
        - Momentum: {analysis['momentum']['value']:.1%} ({analysis['momentum']['signal']})
        - Support Level: ₹{analysis['support']:.2f}
        - Resistance Level: ₹{analysis['resistance']:.2f}"""
    
    elif intent == "ask_stock_volume":
        volume = stock_handler.df['Volume'].iloc[-1]
        return f"The latest trading volume is {volume:,} shares."
    
    elif intent == "ask_financial_advice":
        return responses["ask_financial_advice"][0]
    
    return random.choice(responses["default"])


# Visualization class using Plotly
class StreamlitVisualization:
    def __init__(self, df):
        self.df = df
    
    def create_price_chart(self):
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Close Price',
            line=dict(color='#1f77b4'), 
            hovertemplate='%{y:.2f}'  # Tooltip formatting
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA_20'],
            name='20-day MA',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA_50'],
            name='50-day MA',
            line=dict(color='green', dash='dash')
        ))
        
        fig.update_layout(
            title={'text': 'Price Chart with Moving Averages', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_dark',  
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)  # Reducing margins
        )
        
        return fig

    def create_technical_chart(self):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price with Bollinger Bands', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Price and Bollinger Bands
        fig.add_trace(go.Scatter(
            x=self.df.index, 
            y=self.df['Close'],
            name='Close Price',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['BB_upper'],
            name='Upper BB',
            line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['BB_lower'],
            name='Lower BB',
            line=dict(color='green', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_dark',  # Changed to dark theme
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

# Stock data handler
class StreamlitStockDataHandler:
    def __init__(self, symbol="RELIANCE.NS", period="1y"):
        stock = yf.Ticker(symbol)
        self.df = stock.history(period=period)
        self.symbol = symbol
        
        self.calculate_indicators()
        self.visualization = StreamlitVisualization(self.df)
    
    def calculate_indicators(self):
        # Calculate indicators
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        self.df['BB_middle'] = self.df['Close'].rolling(window=20).mean()
        std = self.df['Close'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (std * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (std * 2)
        
        self.df['Momentum'] = self.df['Close'].pct_change(periods=10)
    
    def get_current_price(self):
        return self.df['Close'].iloc[-1]
    
    def get_price_change(self):
        return ((self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]) / 
                self.df['Close'].iloc[-2] * 100)
    
    def get_technical_analysis(self):
        current_rsi = self.df['RSI'].iloc[-1]
        current_momentum = self.df['Momentum'].iloc[-1]
        
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        momentum_signal = ("Strong Upward" if current_momentum > 0.02 
                         else "Strong Downward" if current_momentum < -0.02 
                         else "Neutral")
        
        support = self.df['Low'].tail(30).min()
        resistance = self.df['High'].tail(30).max()
        
        return {
            'rsi': {'value': current_rsi, 'signal': rsi_signal},
            'momentum': {'value': current_momentum, 'signal': momentum_signal},
            'support': support,
            'resistance': resistance
        }

# Streamlit app
def main():
    # Add a header with a logo or branding image
    st.title("📈 Stock Analysis Chatbot")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'stock_handler' not in st.session_state:
        st.session_state.stock_handler = StreamlitStockDataHandler()
    
    # Sidebar for input
    st.sidebar.title("⚙️ Settings")
    symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS", help="Enter stock symbol (e.g., AAPL for Apple Inc.)")
    period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.sidebar.button("Update Stock Data"):
        st.session_state.stock_handler = StreamlitStockDataHandler(symbol, period)
        st.session_state.messages = []  # Clear chat history
    
    # Display stock metrics
    st.write("### Stock Information")
    current_price = st.session_state.stock_handler.get_current_price()
    price_change = st.session_state.stock_handler.get_price_change()
    
    # Add columns with icons
    col1, col2 = st.columns(2)
    col1.metric("📊 Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f}%")
    col2.metric("📅 Time Period", period)
    
    # Display tabs for charts
    tab1, tab2 = st.tabs(["📈 Price Chart", "📉 Technical Analysis"])
    
    with tab1:
        st.plotly_chart(st.session_state.stock_handler.visualization.create_price_chart(), use_container_width=True)
    
    with tab2:
        st.plotly_chart(st.session_state.stock_handler.visualization.create_technical_chart(), use_container_width=True)
    
    # Chat interface
    st.markdown("---")
    st.subheader("💬 Chat with the Bot")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input field for chat
    if prompt := st.chat_input("Ask about the stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Determine intent based on user input
        intent = get_intent(prompt)
        
        # Generate response based on intent
        if intent in ["ask_stock_price", "ask_stock_trend", "ask_stock_volume", "ask_financial_advice"]:
            response = generate_stock_response(st.session_state.stock_handler, intent)
        else:
            response = random.choice(responses.get(intent, responses["default"]))

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
