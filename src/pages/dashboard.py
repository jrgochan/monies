import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

from src.utils.auth import require_login
from src.models.database import SessionLocal, Wallet, Balance, Transaction
from src.api.exchanges import get_current_prices

def load_user_wallets(user_id):
    """Load user's wallets from database"""
    db = SessionLocal()
    try:
        wallets = db.query(Wallet).filter(Wallet.user_id == user_id).all()
        return wallets
    finally:
        db.close()

def load_wallet_balances(wallet_id):
    """Load wallet balances from database"""
    db = SessionLocal()
    try:
        balances = db.query(Balance).filter(Balance.wallet_id == wallet_id).all()
        return balances
    finally:
        db.close()

def get_portfolio_value(wallets):
    """Calculate total portfolio value with current prices"""
    total_value = 0
    portfolio = {}
    all_currencies = set()
    
    # Get balances for each wallet
    for wallet in wallets:
        balances = load_wallet_balances(wallet.id)
        for balance in balances:
            if balance.currency not in portfolio:
                portfolio[balance.currency] = 0
            portfolio[balance.currency] += balance.amount
            all_currencies.add(balance.currency)
    
    # Get current prices (in demo, we'll simulate this)
    # In production, would call: prices = get_current_prices("binance", list(all_currencies))
    prices = {
        "BTC": 69420.0,
        "ETH": 3500.0,
        "ADA": 0.45,
        "SOL": 95.0,
        "DOGE": 0.12,
        "USDT": 1.0,
        "USDC": 1.0
    }
    
    # Calculate total value
    for currency, amount in portfolio.items():
        if currency in prices:
            total_value += amount * prices[currency]
    
    return total_value, portfolio, prices

def get_recent_transactions(user_id, limit=5):
    """Get recent transactions for the user"""
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).filter(
            Transaction.user_id == user_id
        ).order_by(Transaction.timestamp.desc()).limit(limit).all()
        return transactions
    finally:
        db.close()

def get_portfolio_history():
    """Generate mock portfolio history data for demo"""
    # In a real app, this would retrieve historical data from database
    # For demo, we'll generate random data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    portfolio_values = [100000]  # Start value
    
    # Generate random daily changes
    for i in range(1, len(dates)):
        daily_change = random.uniform(-0.03, 0.03)  # -3% to +3%
        new_value = portfolio_values[-1] * (1 + daily_change)
        portfolio_values.append(new_value)
    
    return pd.DataFrame({
        'date': dates,
        'value': portfolio_values
    })

def get_asset_allocation(portfolio, prices):
    """Calculate asset allocation for portfolio"""
    allocation = []
    
    for currency, amount in portfolio.items():
        if currency in prices:
            value = amount * prices[currency]
            allocation.append({
                'currency': currency,
                'amount': amount,
                'value': value
            })
    
    return allocation

def show_dashboard():
    """Display the dashboard page"""
    # Require login
    user = require_login()
    
    # Add page title
    st.subheader("Portfolio Dashboard")
    
    # Get wallets and calculate portfolio value
    wallets = load_user_wallets(user['id'])
    
    if not wallets:
        st.info("You don't have any wallets configured yet. Go to the Wallets page to add one.")
        return
    
    portfolio_value, portfolio, prices = get_portfolio_value(wallets)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Portfolio Value", 
            value=f"${portfolio_value:,.2f}",
            delta=f"{random.uniform(-2.0, 5.0):.2f}%"  # Mock daily change
        )
    
    with col2:
        st.metric(
            label="Crypto Assets", 
            value=f"{len(portfolio)}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Connected Wallets", 
            value=f"{len(wallets)}",
            delta=None
        )
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Performance", "Allocation", "Recent Activity"])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        # Get portfolio history
        portfolio_history = get_portfolio_history()
        
        # Plot line chart
        fig = px.line(
            portfolio_history, 
            x='date', 
            y='value',
            title='30-Day Portfolio Value'
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Asset Allocation")
        
        # Get asset allocation
        allocation = get_asset_allocation(portfolio, prices)
        
        # Create a pie chart
        labels = [item['currency'] for item in allocation]
        values = [item['value'] for item in allocation]
        
        fig = px.pie(
            values=values,
            names=labels,
            title='Portfolio Allocation by Asset'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display allocation table
        allocation_df = pd.DataFrame(allocation)
        allocation_df['value'] = allocation_df['value'].map('${:,.2f}'.format)
        
        st.dataframe(
            allocation_df.rename(columns={
                'currency': 'Currency',
                'amount': 'Amount',
                'value': 'Value (USD)'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Recent Transactions")
        
        # Get recent transactions
        transactions = get_recent_transactions(user['id'])
        
        if transactions:
            for tx in transactions:
                with st.container(border=True):
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        if tx.transaction_type == 'buy':
                            st.markdown("🟢 **BUY**")
                        elif tx.transaction_type == 'sell':
                            st.markdown("🔴 **SELL**")
                        elif tx.transaction_type == 'send':
                            st.markdown("⬆️ **SEND**")
                        elif tx.transaction_type == 'receive':
                            st.markdown("⬇️ **RECEIVE**")
                    
                    with cols[1]:
                        st.markdown(f"**{tx.amount} {tx.currency}**")
                        if tx.price:
                            st.caption(f"Price: ${tx.price:.2f}")
                        if tx.notes:
                            st.caption(f"Note: {tx.notes}")
                    
                    with cols[2]:
                        st.markdown(f"{tx.timestamp.strftime('%Y-%m-%d')}")
                        st.caption(f"Status: {tx.status}")
        else:
            st.info("No recent transactions found.")
    
    # Display AI insights
    st.markdown("---")
    st.subheader("AI Market Insights")
    
    with st.container(border=True):
        insight_col1, insight_col2 = st.columns([3, 1])
        
        with insight_col1:
            st.markdown("### Today's Market Summary")
            st.markdown("""
            The cryptocurrency market is showing **positive momentum** with Bitcoin leading the charge, 
            up 2.3% in the last 24 hours. Ethereum follows with a 1.8% gain. Overall market sentiment
            appears bullish based on social media sentiment and trading volume across major exchanges.
            
            Tech stocks are also performing well today, with the NASDAQ up 1.2%. There's particular 
            strength in semiconductor and AI-related companies.
            """)
        
        with insight_col2:
            st.markdown("### Trending Assets")
            st.markdown("1. 📈 **BTC** +2.3%")
            st.markdown("2. 📈 **ETH** +1.8%")
            st.markdown("3. 📉 **ADA** -0.5%")
            st.markdown("4. 📈 **NVDA** +3.2%")
    
    # Add a "Refresh" button at the bottom
    if st.button("Refresh Data"):
        st.experimental_rerun()