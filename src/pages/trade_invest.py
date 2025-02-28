import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime

from src.utils.auth import require_login
from src.models.database import SessionLocal, Wallet, Balance, Transaction
from src.utils.security import get_api_key
from src.api.exchanges import place_order, get_current_prices

def get_user_wallets(user_id):
    """Get all wallets for a user"""
    db = SessionLocal()
    try:
        wallets = db.query(Wallet).filter(Wallet.user_id == user_id).all()
        return wallets
    finally:
        db.close()

def get_wallet_balances(wallet_id):
    """Get balances for a wallet"""
    db = SessionLocal()
    try:
        balances = db.query(Balance).filter(Balance.wallet_id == wallet_id).all()
        return balances
    finally:
        db.close()

def get_available_currencies(user_id):
    """Get all currencies available across user's wallets"""
    db = SessionLocal()
    try:
        # Join Wallet and Balance tables to get currencies
        wallets = db.query(Wallet).filter(Wallet.user_id == user_id).all()
        
        currencies = set()
        for wallet in wallets:
            balances = db.query(Balance).filter(Balance.wallet_id == wallet.id).all()
            for balance in balances:
                currencies.add(balance.currency)
        
        return sorted(list(currencies))
    finally:
        db.close()

def add_transaction(user_id, wallet_id, transaction_type, currency, amount, price=None, notes=None):
    """Add a new transaction"""
    db = SessionLocal()
    try:
        # Create transaction
        transaction = Transaction(
            user_id=user_id,
            wallet_id=wallet_id,
            transaction_type=transaction_type,
            currency=currency,
            amount=amount,
            price=price,
            status="completed",
            notes=notes
        )
        
        db.add(transaction)
        db.commit()
        
        # Update wallet balance
        balance = db.query(Balance).filter(
            Balance.wallet_id == wallet_id,
            Balance.currency == currency
        ).first()
        
        if balance:
            if transaction_type in ["buy", "receive"]:
                balance.amount += amount
            elif transaction_type in ["sell", "send"]:
                balance.amount -= amount
        else:
            # Create new balance if it doesn't exist
            if transaction_type in ["buy", "receive"]:
                balance = Balance(
                    wallet_id=wallet_id,
                    currency=currency,
                    amount=amount
                )
                db.add(balance)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error adding transaction: {str(e)}")
        return False
    finally:
        db.close()

def show_crypto_trading():
    """Show crypto trading section"""
    user = require_login()
    
    st.subheader("Cryptocurrency Trading")
    
    # Get user's wallets
    wallets = get_user_wallets(user['id'])
    
    if not wallets:
        st.warning("You need to add a wallet before trading. Go to the Wallets page to add one.")
        return
    
    # Filter only exchange wallets
    exchange_wallets = [w for w in wallets if w.wallet_type == "exchange"]
    
    if not exchange_wallets:
        st.warning("You need to add an exchange wallet to trade. Go to the Wallets page to add one.")
        return
    
    # Select wallet
    wallet_options = {f"{w.name} ({w.exchange})": w for w in exchange_wallets}
    selected_wallet_name = st.selectbox(
        "Select Exchange Wallet",
        options=list(wallet_options.keys())
    )
    selected_wallet = wallet_options[selected_wallet_name]
    
    # Get wallet balances
    balances = get_wallet_balances(selected_wallet.id)
    
    # Display balances
    if balances:
        st.write("Available Balances:")
        
        # Mock prices for demo
        prices = {
            "BTC": 69420.0,
            "ETH": 3500.0,
            "ADA": 0.45,
            "SOL": 95.0,
            "DOGE": 0.12,
            "USDT": 1.0,
            "USDC": 1.0
        }
        
        # Create a table of balances
        balance_data = []
        for balance in balances:
            price = prices.get(balance.currency, 0)
            usd_value = balance.amount * price
            
            balance_data.append({
                "Currency": balance.currency,
                "Amount": f"{balance.amount:.8f}".rstrip('0').rstrip('.'),
                "Price": f"${price:,.2f}" if price > 0 else "-",
                "Value (USD)": f"${usd_value:,.2f}" if price > 0 else "-"
            })
        
        # Display as dataframe
        balance_df = pd.DataFrame(balance_data)
        st.dataframe(balance_df, hide_index=True, use_container_width=True)
    else:
        st.info("No balances found for this wallet.")
    
    # Trading form
    st.subheader("Place Trade")
    
    with st.form("trading_form"):
        # Trading pair (e.g., BTC/USDT)
        trading_pairs = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT",
            "BTC/USDC", "ETH/USDC", "SOL/USDC", "ADA/USDC", "DOGE/USDC"
        ]
        
        symbol = st.selectbox("Trading Pair", options=trading_pairs)
        
        # Parse base and quote currencies
        base_currency, quote_currency = symbol.split('/')
        
        # Order type
        order_type = st.selectbox("Order Type", options=["market", "limit"])
        
        # Side (buy/sell)
        side = st.selectbox("Side", options=["buy", "sell"])
        
        # Amount
        amount = st.number_input(f"Amount ({base_currency})", min_value=0.0, step=0.001)
        
        # Price (for limit orders)
        price = None
        if order_type == "limit":
            price = st.number_input(f"Price ({quote_currency})", min_value=0.0, step=0.01)
        
        # Show estimated value
        # For demo, use mock price
        mock_price = prices.get(base_currency, 0)
        if mock_price > 0:
            estimated_value = amount * mock_price
            st.write(f"Estimated Value: ${estimated_value:,.2f}")
        
        # Submit button
        submitted = st.form_submit_button("Place Order")
        
        if submitted:
            if amount <= 0:
                st.error("Amount must be greater than 0")
            elif order_type == "limit" and (price is None or price <= 0):
                st.error("Price must be greater than 0 for limit orders")
            else:
                # Get API keys
                db = SessionLocal()
                api_key, api_secret = get_api_key(db, user['id'], selected_wallet.exchange)
                db.close()
                
                if not api_key or not api_secret:
                    st.error("API keys not found for this exchange")
                else:
                    # Place order
                    with st.spinner("Placing order..."):
                        # In a real app, this would call the exchange API
                        # For demo, we'll simulate it
                        
                        """
                        # Actual API call would look like this:
                        result = place_order(
                            exchange_name=selected_wallet.exchange,
                            api_key=api_key,
                            api_secret=api_secret,
                            symbol=symbol,
                            order_type=order_type,
                            side=side,
                            amount=amount,
                            price=price
                        )
                        """
                        
                        # Simulate successful order
                        result = {
                            'success': True,
                            'order_id': '12345678',
                            'message': 'Order placed successfully'
                        }
                        
                        if result['success']:
                            # Record the transaction
                            notes = f"{side.upper()} {symbol} via {selected_wallet.exchange} (Order ID: {result['order_id']})"
                            
                            transaction_type = side  # 'buy' or 'sell'
                            
                            if add_transaction(
                                user_id=user['id'],
                                wallet_id=selected_wallet.id,
                                transaction_type=transaction_type,
                                currency=base_currency,
                                amount=amount if side == 'buy' else -amount,
                                price=price if order_type == 'limit' else mock_price,
                                notes=notes
                            ):
                                st.success(f"Order placed successfully! Order ID: {result['order_id']}")
                            else:
                                st.warning("Order placed on exchange, but failed to record in local database.")
                        else:
                            st.error(f"Error placing order: {result['message']}")

def show_etf_investing():
    """Show ETF investing section"""
    user = require_login()
    
    st.subheader("ETF Investing")
    
    # ETF search box
    etf_options = ["SPY", "QQQ", "VTI", "VOO", "ARKK", "XLK", "VGT", "XLV", "VHT", "XLF", "VFH"]
    etf_ticker = st.selectbox("Select ETF", options=etf_options)
    
    if etf_ticker:
        try:
            # Get ETF info
            etf = yf.Ticker(etf_ticker)
            info = etf.info
            
            # Display ETF info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(info.get('shortName', etf_ticker))
                st.caption(info.get('category', ''))
                
                # Description
                if 'longBusinessSummary' in info:
                    st.markdown(info['longBusinessSummary'][:500] + '...')
                
                # Key stats
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Current Price", f"${info.get('regularMarketPrice', 0):,.2f}")
                
                with stats_col2:
                    st.metric(
                        "YTD Return", 
                        f"{info.get('ytdReturn', 0) * 100:.2f}%" if 'ytdReturn' in info else "N/A"
                    )
                
                with stats_col3:
                    st.metric(
                        "Expense Ratio", 
                        f"{info.get('expenseRatio', 0) * 100:.2f}%" if 'expenseRatio' in info else "N/A"
                    )
            
            with col2:
                # Show price chart
                hist = etf.history(period='1y')
                
                fig = px.line(
                    hist, 
                    y='Close',
                    title=f"{etf_ticker} Price (1 Year)"
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Investment form
            st.subheader("Invest in this ETF")
            
            with st.form("etf_invest_form"):
                amount_usd = st.number_input("Investment Amount (USD)", min_value=0.0, step=100.0)
                
                # Source of funds
                fund_source = st.radio(
                    "Source of Funds",
                    options=["Existing Crypto", "External (Brokerage)"],
                    horizontal=True
                )
                
                if fund_source == "Existing Crypto":
                    # Get available currencies
                    currencies = get_available_currencies(user['id'])
                    funding_currency = st.selectbox("Source Currency", options=currencies)
                    
                    # Mock exchange rate
                    exchange_rate = 1.0
                    if funding_currency == "BTC":
                        exchange_rate = 69420.0
                    elif funding_currency == "ETH":
                        exchange_rate = 3500.0
                    
                    # Calculate equivalent amount
                    if exchange_rate > 0:
                        crypto_amount = amount_usd / exchange_rate
                        st.write(f"Equivalent: {crypto_amount:.8f} {funding_currency}")
                
                notes = st.text_area("Investment Notes (optional)")
                
                submitted = st.form_submit_button("Invest")
                
                if submitted:
                    if amount_usd <= 0:
                        st.error("Investment amount must be greater than 0")
                    else:
                        if fund_source == "External (Brokerage)":
                            # Simulate successful investment through external brokerage
                            st.success(f"Investment order for ${amount_usd:,.2f} in {etf_ticker} has been submitted to your brokerage.")
                            st.info("Note: In a real app, this would connect to a brokerage API to place the order.")
                        else:
                            # Simulate converting crypto to USD and investing
                            st.success(f"Successfully converted {crypto_amount:.8f} {funding_currency} to ${amount_usd:,.2f} and invested in {etf_ticker}.")
                            st.info("Note: In a real app, this would execute trades on the exchange and brokerage.")
                            
                            # Record as a transaction
                            wallets = get_user_wallets(user['id'])
                            if wallets:
                                # Just use the first wallet for demo purposes
                                wallet_id = wallets[0].id
                                
                                add_transaction(
                                    user_id=user['id'],
                                    wallet_id=wallet_id,
                                    transaction_type="sell",
                                    currency=funding_currency,
                                    amount=crypto_amount,
                                    price=exchange_rate,
                                    notes=f"Converted to USD for investing in {etf_ticker} ETF. {notes}"
                                )
        
        except Exception as e:
            st.error(f"Error loading ETF data: {str(e)}")

def show_rebalance():
    """Show portfolio rebalance section"""
    user = require_login()
    
    st.subheader("Portfolio Rebalancing")
    
    # Get user's assets
    currencies = get_available_currencies(user['id'])
    
    if not currencies:
        st.warning("You don't have any assets to rebalance. Add wallets and funds first.")
        return
    
    # Create a form for target allocation
    with st.form("rebalance_form"):
        st.write("Set your target allocation percentages:")
        
        # Create sliders for each currency
        allocations = {}
        for currency in currencies:
            allocations[currency] = st.slider(
                f"{currency} Allocation",
                min_value=0, 
                max_value=100, 
                value=100 // len(currencies),  # Equal allocation by default
                key=f"allocation_{currency}"
            )
        
        # Validate total = 100%
        total_allocation = sum(allocations.values())
        if total_allocation != 100:
            st.warning(f"Total allocation: {total_allocation}% (should be 100%)")
        
        # Rebalance frequency
        frequency = st.selectbox(
            "Rebalancing Frequency",
            options=["Manual", "Monthly", "Quarterly"]
        )
        
        submitted = st.form_submit_button("Set Allocation")
        
        if submitted:
            if total_allocation != 100:
                st.error("Total allocation must equal 100%")
            else:
                st.success("Portfolio allocation targets have been set!")
                
                # Display allocation as a pie chart
                fig = px.pie(
                    values=list(allocations.values()),
                    names=list(allocations.keys()),
                    title='Target Portfolio Allocation'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show next steps
                st.info(f"Your portfolio will be rebalanced {frequency.lower()}. You can also trigger a manual rebalance below.")
                
                # Provide rebalance button
                if st.button("Rebalance Now"):
                    with st.spinner("Analyzing portfolio and calculating trades..."):
                        # In a real app, this would calculate required trades
                        # to achieve the target allocation
                        
                        # Simulate rebalance process
                        st.success("Portfolio rebalanced successfully!")
                        st.write("The following trades were executed:")
                        
                        # Mock rebalancing trades
                        trades = [
                            {"type": "sell", "currency": "BTC", "amount": 0.01, "value": "$700"},
                            {"type": "buy", "currency": "ETH", "amount": 0.2, "value": "$700"}
                        ]
                        
                        # Display in a table
                        trade_df = pd.DataFrame(trades)
                        st.dataframe(trade_df, hide_index=True)

def show_trade_invest():
    """Display the Trade/Invest page"""
    # Require login
    user = require_login()
    
    # Add page title
    st.title("Trade & Invest")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "Cryptocurrency Trading", 
        "ETF Investing", 
        "Portfolio Rebalancing"
    ])
    
    with tab1:
        show_crypto_trading()
    
    with tab2:
        show_etf_investing()
    
    with tab3:
        show_rebalance()