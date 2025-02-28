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
    
    # Get current prices from exchange API (using Binance.US)
    prices = get_current_prices("binanceus", list(all_currencies))
    
    # Check if we have an error in the prices result
    if "error" in prices:
        return 0, portfolio, prices
    
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

def get_portfolio_history(wallets=None):
    """Get portfolio history data. This function returns an error flag
    if it cannot connect to external data sources.
    """
    # Initialize error flag
    error = False
    error_message = ""
    
    # Create a valid portfolio structure
    portfolio = {}
    all_currencies = set()
    current_value = 0
    
    if wallets:
        try:
            # Get balances for each wallet
            for wallet in wallets:
                balances = load_wallet_balances(wallet.id)
                for balance in balances:
                    if balance.currency not in portfolio:
                        portfolio[balance.currency] = 0
                    portfolio[balance.currency] += balance.amount
                    all_currencies.add(balance.currency)
            
            # Only proceed if we have any currencies
            if all_currencies:
                # Get current prices
                prices = get_current_prices("binanceus", list(all_currencies))
                
                # Check if prices have an error
                if "error" in prices:
                    error = True
                    error_message = prices["error"]
                    # Return error dataframe
                    return pd.DataFrame({
                        'date': [datetime.now()],
                        'value': [0],
                        'error': [True],
                        'error_message': [error_message]
                    })
                
                # Calculate current portfolio value
                for currency, amount in portfolio.items():
                    if currency in prices:
                        current_value += amount * prices[currency]
        except Exception as e:
            import logging
            logging.error(f"Error generating portfolio history: {str(e)}")
            error = True
            error_message = f"Failed to retrieve portfolio data: {str(e)}"
            # Return error dataframe
            return pd.DataFrame({
                'date': [datetime.now()],
                'value': [0],
                'error': [True],
                'error_message': [error_message]
            })
    
    # If we have no wallets or couldn't calculate value
    if not wallets or current_value <= 0:
        return pd.DataFrame({
            'date': [datetime.now()],
            'value': [0],
            'error': [True],
            'error_message': ["No portfolio data available or cannot connect to price source."]
        })
    
    # At this point we have the current value but can't produce historical data
    # Return a dataframe with just the current date and value, plus error indicators
    return pd.DataFrame({
        'date': [datetime.now()],
        'value': [current_value],
        'error': [True],
        'error_message': ["Cannot retrieve historical data. Please check your network connection."]
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
    
    # Check if we have an error in the prices
    if "error" in prices:
        st.error(prices["error"])
        st.warning("Cannot retrieve current portfolio data. Please check your network connection and try again.")
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Display portfolio value or error message
        if "error" in prices:
            st.metric(
                label="Total Portfolio Value", 
                value="Error",
                delta=None
            )
        else:
            st.metric(
                label="Total Portfolio Value", 
                value=f"${portfolio_value:,.2f}",
                delta=None  # Removed the random delta
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
        
        # Get portfolio history based on current wallet data
        portfolio_history = get_portfolio_history(wallets)
        
        # Check if we have an error in the portfolio history
        if 'error' in portfolio_history.columns and portfolio_history['error'].iloc[0]:
            # Display error message instead of chart
            st.error(portfolio_history['error_message'].iloc[0])
            st.warning("Cannot display portfolio performance chart. Please check your network connection and try again.")
        else:
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
        
        # Check if we have an error in the prices
        if "error" in prices:
            st.error(prices["error"])
            st.warning("Cannot display asset allocation. Please check your network connection and try again.")
        else:
            # Get asset allocation
            allocation = get_asset_allocation(portfolio, prices)
            
            # Create a pie chart if we have allocation data
            if allocation:
                labels = [item['currency'] for item in allocation]
                values = [item['value'] for item in allocation]
                
                if labels and values:
                    fig = px.pie(
                        values=values,
                        names=labels,
                        title='Portfolio Allocation by Asset'
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for allocation chart. Check your wallet balances.")
            else:
                st.info("No asset allocation data available. Try adding cryptocurrency to your wallets.")
        
        # Display allocation table
        if allocation:
            allocation_df = pd.DataFrame(allocation)
            # Check if the dataframe actually has the 'value' column before formatting
            if 'value' in allocation_df.columns:
                allocation_df['value'] = allocation_df['value'].map('${:,.2f}'.format)
            
            # Rename columns, ensure all expected columns exist
            column_mapping = {}
            if 'currency' in allocation_df.columns:
                column_mapping['currency'] = 'Currency'
            if 'amount' in allocation_df.columns:
                column_mapping['amount'] = 'Amount'
            if 'value' in allocation_df.columns:
                column_mapping['value'] = 'Value (USD)'
            
            st.dataframe(
                allocation_df.rename(columns=column_mapping),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No asset allocation data available. Try adding cryptocurrency to your wallets.")
    
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