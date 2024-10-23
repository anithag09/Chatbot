import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import datetime
import yfinance as yf

# Utility classes and functions

class FinancialIntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifier:
    def __init__(self, num_labels: int, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=num_labels
        )
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def predict(self, text: str) -> Tuple[int, float]:
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, prediction = torch.max(outputs.logits, dim=1)
            probability = torch.nn.functional.softmax(outputs.logits, dim=1)

        return prediction.item(), probability.max().item()

class RiskAssessmentModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def predict_risk_profile(self, features: pd.DataFrame) -> Tuple[str, float]:
        X_scaled = self.scaler.transform(features)
        risk_score = self.model.predict_proba(X_scaled)[0]
        risk_level = self.model.predict(X_scaled)[0]
        
        return risk_level, max(risk_score)

class GoalBasedInvestment:
    def __init__(self):
        self.monte_carlo_sims = 1000

    def simulate_returns(self, initial_investment: float, monthly_contribution: float,
                        years: int, expected_return: float, volatility: float) -> pd.DataFrame:
        periods = years * 12
        returns = np.random.normal(expected_return/12, volatility/np.sqrt(12), 
                                 size=(self.monte_carlo_sims, periods))
        
        portfolio_values = np.zeros((self.monte_carlo_sims, periods))
        portfolio_values[:, 0] = initial_investment
        
        for i in range(1, periods):
            portfolio_values[:, i] = (portfolio_values[:, i-1] * (1 + returns[:, i]) + 
                                    monthly_contribution)
        
        return pd.DataFrame(portfolio_values)

    def analyze_goal_achievement(self, current_portfolio: float, goal_amount: float,
                               time_horizon: int, risk_profile: str, monthly_contribution: float) -> Dict:
        risk_params = {
            'conservative': {'return': 0.06, 'volatility': 0.08},
            'moderate': {'return': 0.08, 'volatility': 0.12},
            'aggressive': {'return': 0.10, 'volatility': 0.16}
        }
        
        params = risk_params.get(risk_profile.lower(), risk_params['moderate'])
        
        simulations = self.simulate_returns(
            initial_investment=current_portfolio,
            monthly_contribution=monthly_contribution,
            years=time_horizon,
            expected_return=params['return'],
            volatility=params['volatility']
        )
        
        final_values = simulations.iloc[:, -1]
        probability_success = (final_values >= goal_amount).mean()
        
        return {
            'probability_success': probability_success,
            'median_final_value': final_values.median(),
            'worst_case': final_values.quantile(0.05),
            'best_case': final_values.quantile(0.95),
            'current_portfolio': current_portfolio,
            'goal_amount': goal_amount,
            'simulations': simulations
        }

class FinancialAdvisor:
    def __init__(self):
        self.intent_classifier = IntentClassifier(num_labels=7)
        self.risk_model = RiskAssessmentModel()
        self.goal_analyzer = GoalBasedInvestment()
        self.knowledge_base = self._load_knowledge_base()

    @staticmethod
    def _load_knowledge_base() -> Dict:
        return {
            "investment_products": {
                "stocks": {"risk": "high", "min_horizon": 5},
                "bonds": {"risk": "low", "min_horizon": 2},
                "mutual_funds": {"risk": "moderate", "min_horizon": 3}
            },
            "risk_profiles": {
                "conservative": {"equity": 0.3, "bonds": 0.6, "cash": 0.1},
                "moderate": {"equity": 0.6, "bonds": 0.3, "cash": 0.1},
                "aggressive": {"equity": 0.8, "bonds": 0.15, "cash": 0.05}
            }
        }

    def process_query(self, user_query: str, user_context: Dict) -> Dict:
        intent, confidence = self.intent_classifier.predict(user_query)
        
        if intent == 0:  # risk_assessment
            risk_profile = self._assess_risk(user_context)
            response = self._generate_risk_recommendation(risk_profile)
        elif intent == 1:  # goal_planning
            goal_analysis = self._analyze_financial_goals(user_context)
            response = self._generate_goal_recommendations(goal_analysis)
        else:
            response = self._generate_general_advice(intent, user_context)
            
        return {
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _assess_risk(self, user_context: Dict) -> str:
        features = pd.DataFrame([{
            'age': user_context['age'],
            'income': user_context['income'],
            'dependents': user_context['dependents'],
            'savings': user_context['savings'],
            'investment_experience': user_context['investment_experience']
        }])
        
        risk_level, confidence = self.risk_model.predict_risk_profile(features)
        return risk_level

    def _analyze_financial_goals(self, user_context: Dict) -> Dict:
        return self.goal_analyzer.analyze_goal_achievement(
            current_portfolio=user_context['current_portfolio'],
            goal_amount=user_context['goal_amount'],
            time_horizon=user_context['time_horizon'],
            risk_profile=user_context['risk_profile'],
            monthly_contribution=user_context['monthly_contribution']
        )

    def _generate_risk_recommendation(self, risk_profile: str) -> str:
        portfolio_allocation = self.knowledge_base['risk_profiles'][risk_profile]
        
        return f"Based on your risk profile ({risk_profile}), here's your recommended portfolio allocation:\n" + \
               f"- Equity: {portfolio_allocation['equity']*100}%\n" + \
               f"- Bonds: {portfolio_allocation['bonds']*100}%\n" + \
               f"- Cash: {portfolio_allocation['cash']*100}%"

    def _generate_goal_recommendations(self, goal_analysis: Dict) -> str:
        return f"Goal Analysis Results:\n" + \
               f"- Probability of Success: {goal_analysis['probability_success']*100:.1f}%\n" + \
               f"- Median Expected Value: ${goal_analysis['median_final_value']:,.2f}\n" + \
               f"- Worst Case (5th percentile): ${goal_analysis['worst_case']:,.2f}\n" + \
               f"- Best Case (95th percentile): ${goal_analysis['best_case']:,.2f}"

    def _generate_general_advice(self, intent: int, user_context: Dict) -> str:
        advice_map = {
            2: "When considering investing in stocks, it's important to diversify your portfolio and consider your risk tolerance. Long-term investing often yields better results.",
            3: "The best investment strategy depends on your financial goals, risk tolerance, and time horizon. A balanced portfolio often includes a mix of stocks, bonds, and other assets.",
            4: "Saving for retirement involves consistent contributions, taking advantage of employer matches in 401(k) plans, and adjusting your strategy as you age.",
            5: "Budgeting is key to financial health. Try the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment.",
            6: "When managing debt, prioritize high-interest debt first. Consider consolidation for better rates, and always pay more than the minimum when possible."
        }
        return advice_map.get(intent, "I'm not sure how to advise on that topic. Could you please rephrase or ask about something else?")

# Streamlit App

def create_portfolio_pie_chart(allocations: Dict[str, float]):
    fig = go.Figure(data=[go.Pie(
        labels=list(allocations.keys()),
        values=[v * 100 for v in allocations.values()],
        hole=.3
    )])
    fig.update_layout(
        title="Recommended Portfolio Allocation",
        showlegend=True,
        height=400
    )
    return fig

def create_goal_simulation_chart(goal_analysis: Dict):
    fig = go.Figure()
    
    x = list(range(len(goal_analysis['simulations'].iloc[0])))
    
    # Plot all simulations
    for i in range(min(100, len(goal_analysis['simulations']))):
        fig.add_trace(go.Scatter(
            x=x, 
            y=goal_analysis['simulations'].iloc[i],
            mode='lines',
            line=dict(color='rgba(0,100,80,0.1)'),
            showlegend=False
        ))
    
    # Add median line
    median_values = goal_analysis['simulations'].median()
    fig.add_trace(go.Scatter(
        x=x,
        y=median_values,
        mode='lines',
        line=dict(color='rgb(0,100,80)', width=4),
        name='Median Path'
    ))
    
    # Add goal line
    fig.add_trace(go.Scatter(
        x=[x[0], x[-1]],
        y=[goal_analysis['goal_amount'], goal_analysis['goal_amount']],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Goal Amount'
    ))
    
    fig.update_layout(
        title="Investment Goal Projection",
        xaxis_title="Months",
        yaxis_title="Portfolio Value ($)",
        showlegend=True,
        height=500
    )
    
    return fig

def fetch_stock_data(symbol: str, period: str = '1y'):
    stock = yf.Ticker(symbol)
    history = stock.history(period=period)
    return history

def create_stock_chart(data: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    return fig

@st.cache_resource
def load_advisor():
    return FinancialAdvisor()

def main():
    st.set_page_config(page_title="AI Financial Advisor", layout="wide")
    
    st.title("🤖 AI Financial Advisor")
    st.markdown("""
    Your personal AI-powered financial advisor. Get personalized investment advice,
    risk assessment, goal planning assistance, and more.
    """)
    
    advisor = load_advisor()
    
    # Sidebar for user information
    st.sidebar.header("Your Profile")
    age = st.sidebar.slider("Age", 18, 100, 35)
    income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 80000, step=1000)
    dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
    savings = st.sidebar.number_input("Current Savings ($)", 0, 1000000, 50000, step=1000)
    investment_experience = st.sidebar.slider("Investment Experience (1-10)", 1, 10, 3)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment", "Goal Planning", "Stock Analysis", "Chat with Advisor"])
    
    # User context dictionary
    user_context = {
        "age": age,
        "income": income,
        "dependents": dependents,
        "savings": savings,
        "investment_experience": investment_experience,
        "current_portfolio": savings,
        "goal_amount": 500000,
        "time_horizon": 20,
        "risk_profile": "moderate",
        "monthly_contribution": 500
    }
    
    # Tab 1: Risk Assessment
    with tab1:
        st.header("Risk Profile Assessment")
        
        if st.button("Analyze Risk Profile"):
            risk_profile = advisor._assess_risk(user_context)
            st.write(f"Your risk profile is: **{risk_profile.capitalize()}**")
            
            recommendation = advisor._generate_risk_recommendation(risk_profile)
            st.write(recommendation)
            
            # Create and display portfolio allocation pie chart
            allocations = advisor.knowledge_base['risk_profiles'][risk_profile]
            fig = create_portfolio_pie_chart(allocations)
            st.plotly_chart(fig)
    
    # Tab 2: Goal Planning
    with tab2:
        st.header("Financial Goal Planning")
        
        user_context['goal_amount'] = st.number_input("Goal Amount ($)", 10000, 10000000, 500000, step=10000)
        user_context['time_horizon'] = st.slider("Time Horizon (years)", 1, 40, 20)
        user_context['monthly_contribution'] = st.number_input("Monthly Contribution ($)", 0, 10000, 500, step=100)
        
        if st.button("Analyze Goal"):
            goal_analysis = advisor._analyze_financial_goals(user_context)
            recommendations = advisor._generate_goal_recommendations(goal_analysis)
            st.write(recommendations)
            
            # Create and display goal simulation chart
            fig = create_goal_simulation_chart(goal_analysis)
            st.plotly_chart(fig)
    
    # Tab 3: Stock Analysis
    with tab3:
        st.header("Stock Analysis")
        
        stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
        period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        
        if st.button("Analyze Stock"):
            with st.spinner("Fetching stock data..."):
                stock_data = fetch_stock_data(stock_symbol, period)
                
            if not stock_data.empty:
                fig = create_stock_chart(stock_data, stock_symbol)
                st.plotly_chart(fig)
                
                st.write("Recent Stock Data:")
                st.dataframe(stock_data.tail())
            else:
                st.error("Unable to fetch stock data. Please check the stock symbol and try again.")
    
    # Tab 4: Chat with Advisor
    with tab4:
        st.header("Chat with AI Financial Advisor")
        
        user_query = st.text_input("Ask your financial question:")
        
        if st.button("Get Advice"):
            if user_query:
                response = advisor.process_query(user_query, user_context)
                st.write("AI Advisor:", response['response'])
                st.write(f"Confidence: {response['confidence']:.2f}")
            else:
                st.warning("Please enter a question to get advice.")

if __name__ == "__main__":
    main()