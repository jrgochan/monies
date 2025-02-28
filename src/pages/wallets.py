import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.api.exchanges import (
    get_exchange_client,
    get_supported_exchanges,
    get_transaction_history,
    get_wallet_balance,
)
from src.models.database import ApiKey, Balance, SessionLocal, Transaction, Wallet
from src.utils.auth import require_login
from src.utils.security import get_api_key, store_api_key


def get_user_wallets(user_id):
    """Get all wallets for a user"""
    db = SessionLocal()
    try:
        wallets = db.query(Wallet).filter(Wallet.user_id == user_id).all()
        return wallets
    finally:
        db.close()


def add_wallet(user_id, name, wallet_type, exchange=None, address=None):
    """Add a new wallet for a user"""
    db = SessionLocal()
    try:
        # Create new wallet
        wallet = Wallet(
            user_id=user_id,
            name=name,
            wallet_type=wallet_type,
            exchange=exchange,
            address=address,
        )

        db.add(wallet)
        db.commit()
        db.refresh(wallet)

        return wallet
    except Exception as e:
        db.rollback()
        st.error(f"Error adding wallet: {str(e)}")
        return None
    finally:
        db.close()


def update_wallet_balances(wallet_id, balances_data):
    """Update balances for a wallet"""
    db = SessionLocal()
    try:
        # Delete existing balances
        db.query(Balance).filter(Balance.wallet_id == wallet_id).delete()

        # Add new balances
        for currency, data in balances_data.items():
            balance = Balance(
                wallet_id=wallet_id, currency=currency, amount=data["total"]
            )
            db.add(balance)

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error updating balances: {str(e)}")
        return False
    finally:
        db.close()


def delete_wallet(wallet_id):
    """Delete a wallet"""
    db = SessionLocal()
    try:
        # Delete balances first (foreign key constraint)
        db.query(Balance).filter(Balance.wallet_id == wallet_id).delete()

        # Delete wallet
        db.query(Wallet).filter(Wallet.id == wallet_id).delete()

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error deleting wallet: {str(e)}")
        return False
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


def add_transaction(
    user_id, wallet_id, transaction_type, currency, amount, price=None, notes=None
):
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
            notes=notes,
        )

        db.add(transaction)
        db.commit()

        # Update wallet balance
        balance = (
            db.query(Balance)
            .filter(Balance.wallet_id == wallet_id, Balance.currency == currency)
            .first()
        )

        if balance:
            if transaction_type in ["buy", "receive"]:
                balance.amount += amount
            elif transaction_type in ["sell", "send"]:
                balance.amount -= amount
        else:
            # Create new balance if it doesn't exist
            if transaction_type in ["buy", "receive"]:
                balance = Balance(wallet_id=wallet_id, currency=currency, amount=amount)
                db.add(balance)

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error adding transaction: {str(e)}")
        return False
    finally:
        db.close()


def get_wallet_transactions(wallet_id):
    """Get transactions for a wallet"""
    db = SessionLocal()
    try:
        transactions = (
            db.query(Transaction)
            .filter(Transaction.wallet_id == wallet_id)
            .order_by(Transaction.timestamp.desc())
            .limit(50)
            .all()
        )
        return transactions
    finally:
        db.close()


def show_add_wallet_form(user_id):
    """Display form to add a new wallet"""
    with st.expander("Add New Wallet", expanded=False):
        with st.form("add_wallet_form"):
            st.write("Add a new wallet or exchange account")

            name = st.text_input("Wallet Name", placeholder="My Wallet")

            wallet_type = st.selectbox(
                "Wallet Type",
                options=["exchange", "on-chain", "hardware", "mobile", "browser"],
                format_func=lambda x: {
                    "exchange": "Exchange Account",
                    "on-chain": "On-chain Wallet",
                    "hardware": "Hardware Wallet",
                    "mobile": "Mobile Wallet",
                    "browser": "Browser Wallet",
                }.get(x, x.capitalize()),
            )

            if wallet_type == "exchange":
                # Get supported exchanges and put binanceus at the top of the list
                exchanges = get_supported_exchanges()

                # Attempt to reorder to prioritize binanceus
                if "binanceus" in exchanges:
                    exchanges.remove("binanceus")
                    exchanges = ["binanceus"] + exchanges

                # Remove regular binance to avoid confusion
                if "binance" in exchanges:
                    exchanges.remove("binance")

                exchange = st.selectbox("Exchange", options=exchanges, index=0)

                # Show helpful message about Binance.US
                if exchange == "binanceus":
                    st.info(
                        "Using Binance.US which is appropriate for US-based users. Make sure your API keys are from Binance.US, not regular Binance."
                    )

                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")

                address = None
            elif wallet_type in ["hardware", "mobile", "browser"]:
                exchange = None
                api_key = None
                api_secret = None

                wallet_provider = ""
                if wallet_type == "hardware":
                    wallet_provider = st.selectbox(
                        "Hardware Wallet Provider",
                        options=[
                            "Ledger",
                            "Trezor",
                            "KeepKey",
                            "SafePal",
                            "Coldcard",
                            "BitBox",
                            "Ellipal",
                            "Other",
                        ],
                    )
                elif wallet_type == "mobile":
                    wallet_provider = st.selectbox(
                        "Mobile Wallet Provider",
                        options=[
                            "Metamask",
                            "Trust Wallet",
                            "Exodus",
                            "Coinbase Wallet",
                            "Blockchain.com",
                            "Argent",
                            "Binance Wallet",
                            "Rainbow",
                            "Other",
                        ],
                    )
                elif wallet_type == "browser":
                    wallet_provider = st.selectbox(
                        "Browser Wallet Provider",
                        options=[
                            "Metamask",
                            "Brave Wallet",
                            "Coinbase Wallet",
                            "Phantom",
                            "WalletConnect",
                            "Exodus",
                            "Rabby",
                            "Other",
                        ],
                    )

                if wallet_provider == "Other":
                    wallet_provider = st.text_input(
                        "Specify Provider", placeholder="Wallet Provider Name"
                    )

                address = st.text_input("Wallet Address", placeholder="0x...")
                st.info(
                    "Enter the public address for this wallet. This allows you to track balance and transactions."
                )

                # Add a note about wallet provider
                name = (
                    f"{name} ({wallet_provider})"
                    if name and wallet_provider
                    else name or wallet_provider
                )
            else:
                # Original on-chain wallet
                exchange = None
                api_key = None
                api_secret = None

                address = st.text_input("Wallet Address", placeholder="0x...")

            submitted = st.form_submit_button("Add Wallet")

            if submitted:
                if not name:
                    st.error("Please enter a wallet name")
                    return

                if wallet_type == "exchange" and (not api_key or not api_secret):
                    st.error("API Key and Secret are required for exchange accounts")
                    return

                if (
                    wallet_type in ["on-chain", "hardware", "mobile", "browser"]
                    and not address
                ):
                    st.error("Wallet address is required for this wallet type")
                    return

                # Add wallet
                wallet = add_wallet(
                    user_id=user_id,
                    name=name,
                    wallet_type=wallet_type,
                    exchange=exchange,
                    address=address,
                )

                if wallet:
                    # Store API keys if it's an exchange
                    if wallet_type == "exchange":
                        db = SessionLocal()
                        try:
                            store_api_key(db, user_id, exchange, api_key, api_secret)
                        finally:
                            db.close()

                    st.success(f"Wallet '{name}' added successfully!")
                    time.sleep(1)
                    st.rerun()


def show_wallet_card(wallet, user_id, prices=None):
    """Display a wallet card with balances and actions"""
    with st.container(border=True):
        # Header row with wallet name and options
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(wallet.name)
            if wallet.wallet_type == "exchange":
                st.caption(f"Exchange: {wallet.exchange}")
            elif wallet.wallet_type in ["on-chain", "hardware", "mobile", "browser"]:
                wallet_type_display = {
                    "on-chain": "On-chain Wallet",
                    "hardware": "Hardware Wallet",
                    "mobile": "Mobile Wallet",
                    "browser": "Browser Wallet",
                }.get(wallet.wallet_type, wallet.wallet_type)

                st.caption(
                    f"{wallet_type_display}: {wallet.address[:10]}...{wallet.address[-6:]}"
                )
            else:
                st.caption(f"Address: {wallet.address[:10]}...{wallet.address[-6:]}")

        with col2:
            # Add options menu
            option = st.selectbox(
                "Actions",
                options=["Refresh", "Transactions", "Send", "Receive", "Delete"],
                key=f"action_{wallet.id}",
                label_visibility="collapsed",
            )

            if option == "Refresh":
                # For exchange wallets, refresh balances from the exchange
                if wallet.wallet_type == "exchange":
                    with st.spinner("Refreshing balances..."):
                        try:
                            db = SessionLocal()
                            api_key, api_secret = get_api_key(
                                db, user_id, wallet.exchange
                            )
                            db.close()

                            if api_key and api_secret:
                                # Use binanceus for all Binance connections
                                exchange_name = (
                                    "binanceus"
                                    if wallet.exchange.lower() == "binance"
                                    else wallet.exchange
                                )

                                # If it's a Binance account, show info message
                                if wallet.exchange.lower() == "binance":
                                    st.info("Using Binance.US API for compatibility")

                                balances = get_wallet_balance(
                                    exchange_name, api_key, api_secret
                                )
                                if balances:
                                    if update_wallet_balances(wallet.id, balances):
                                        st.success("Balances updated")
                                        time.sleep(1)
                                        st.rerun()
                                else:
                                    st.error(
                                        "Failed to fetch balances. Make sure your API keys are from Binance.US, not regular Binance."
                                    )
                            else:
                                st.error("API keys not found")
                        except Exception as e:
                            st.error(f"Error refreshing balances: {str(e)}")

            elif option == "Delete":
                if st.button(
                    f"Confirm Delete {wallet.name}", key=f"delete_{wallet.id}"
                ):
                    if delete_wallet(wallet.id):
                        st.success(f"Wallet '{wallet.name}' deleted")
                        time.sleep(1)
                        st.rerun()

            elif option == "Transactions":
                st.subheader("Recent Transactions")

                # For exchange wallets, fetch from the exchange API
                if wallet.wallet_type == "exchange":
                    try:
                        db = SessionLocal()
                        api_key, api_secret = get_api_key(db, user_id, wallet.exchange)
                        db.close()

                        if api_key and api_secret:
                            # Use binanceus for all Binance connections
                            exchange_name = (
                                "binanceus"
                                if wallet.exchange.lower() == "binance"
                                else wallet.exchange
                            )

                            # If it's a Binance account, show info message
                            if wallet.exchange.lower() == "binance":
                                st.info("Using Binance.US API for compatibility")

                            with st.spinner("Fetching transactions..."):
                                txs = get_transaction_history(
                                    exchange_name, api_key, api_secret
                                )
                                if txs:
                                    # Create a dataframe for display
                                    df = pd.DataFrame(txs)
                                    df["date"] = pd.to_datetime(
                                        df["timestamp"], unit="ms"
                                    )

                                    # Display transactions
                                    st.dataframe(
                                        df[
                                            [
                                                "date",
                                                "symbol",
                                                "side",
                                                "amount",
                                                "price",
                                                "cost",
                                            ]
                                        ],
                                        hide_index=True,
                                        use_container_width=True,
                                    )
                                else:
                                    st.info("No transactions found")
                        else:
                            st.error("API keys not found")
                    except Exception as e:
                        st.error(f"Error fetching transactions: {str(e)}")

                # For both wallet types, also show transactions from our database
                transactions = get_wallet_transactions(wallet.id)
                if transactions:
                    st.subheader("Recorded Transactions")

                    # Convert to dataframe for easy display
                    tx_data = []
                    for tx in transactions:
                        tx_data.append(
                            {
                                "Date": tx.timestamp.strftime("%Y-%m-%d %H:%M"),
                                "Type": tx.transaction_type.upper(),
                                "Currency": tx.currency,
                                "Amount": tx.amount,
                                "Price": f"${tx.price:.2f}" if tx.price else "-",
                                "Status": tx.status,
                            }
                        )

                    tx_df = pd.DataFrame(tx_data)
                    st.dataframe(tx_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No recorded transactions in database")

            elif option == "Send":
                st.subheader("Send Crypto")

                # Get wallet balances
                balances = get_wallet_balances(wallet.id)

                if not balances:
                    st.warning("No funds available to send")
                else:
                    # Create a form to send crypto
                    with st.form(f"send_form_{wallet.id}"):
                        # Create dropdown with available currencies
                        currency_options = [b.currency for b in balances]
                        currency = st.selectbox("Currency", options=currency_options)

                        # Get current balance for selected currency
                        selected_balance = next(
                            (b for b in balances if b.currency == currency), None
                        )

                        if selected_balance:
                            st.info(f"Available: {selected_balance.amount} {currency}")

                            amount = st.number_input(
                                "Amount to Send",
                                min_value=0.0,
                                max_value=float(selected_balance.amount),
                                step=0.01,
                            )

                            destination = st.text_input("Destination Address")

                            notes = st.text_area("Notes (optional)")

                            submitted = st.form_submit_button("Send")

                            if submitted:
                                if amount <= 0:
                                    st.error("Amount must be greater than 0")
                                elif not destination:
                                    st.error("Destination address is required")
                                else:
                                    # Here would be API call to actually send crypto
                                    # For demo, just record the transaction
                                    if add_transaction(
                                        user_id=user_id,
                                        wallet_id=wallet.id,
                                        transaction_type="send",
                                        currency=currency,
                                        amount=amount,
                                        notes=f"Sent to {destination}. {notes}",
                                    ):
                                        st.success(
                                            f"Sent {amount} {currency} to {destination}"
                                        )
                                        time.sleep(1)
                                        st.rerun()

            elif option == "Receive":
                st.subheader("Receive Crypto")

                if wallet.wallet_type == "exchange":
                    # For exchanges, show deposit addresses
                    st.info(
                        f"Log in to your {wallet.exchange} account to find deposit addresses."
                    )

                    # Could expand to fetch deposit addresses via API
                    st.write("Common deposit addresses on exchanges:")

                    with st.expander("How to find deposit addresses"):
                        st.write(
                            """
                        1. Log in to your exchange account
                        2. Navigate to 'Wallet' or 'Funds' section
                        3. Look for 'Deposit' option
                        4. Select the cryptocurrency you want to deposit
                        5. Copy the deposit address provided
                        """
                        )

                    # Add option to manually record a receive transaction
                    st.subheader("Record a received transaction")

                    with st.form(f"receive_form_{wallet.id}"):
                        currency = st.text_input("Currency")
                        amount = st.number_input(
                            "Amount Received", min_value=0.0, step=0.01
                        )
                        notes = st.text_area("Notes (optional)")

                        submitted = st.form_submit_button("Record")

                        if submitted:
                            if not currency:
                                st.error("Currency is required")
                            elif amount <= 0:
                                st.error("Amount must be greater than 0")
                            else:
                                if add_transaction(
                                    user_id=user_id,
                                    wallet_id=wallet.id,
                                    transaction_type="receive",
                                    currency=currency,
                                    amount=amount,
                                    notes=notes,
                                ):
                                    st.success(
                                        f"Recorded receipt of {amount} {currency}"
                                    )
                                    time.sleep(1)
                                    st.rerun()

                else:
                    # For wallet types with addresses, show QR code and address
                    wallet_type_display = ""
                    wallet_specific_instructions = ""

                    if wallet.wallet_type == "hardware":
                        wallet_type_display = "Hardware Wallet"
                        wallet_specific_instructions = """
                        To receive funds to your hardware wallet:
                        1. Connect your hardware wallet to your computer
                        2. Open the associated wallet app (Ledger Live, Trezor Suite, etc.)
                        3. Follow the instructions to generate a receive address
                        4. Verify the address matches what's shown below
                        """
                    elif wallet.wallet_type == "mobile":
                        wallet_type_display = "Mobile Wallet"
                        wallet_specific_instructions = """
                        To receive funds to your mobile wallet:
                        1. Open your wallet app
                        2. Go to the Receive section
                        3. Verify the address matches what's shown below
                        """
                    elif wallet.wallet_type == "browser":
                        wallet_type_display = "Browser Wallet"
                        wallet_specific_instructions = """
                        To receive funds to your browser wallet:
                        1. Open your browser extension
                        2. Copy your wallet address
                        3. Verify it matches what's shown below
                        """
                    else:
                        wallet_type_display = "On-chain Wallet"

                    # Show wallet specific instructions if available
                    if wallet_specific_instructions:
                        with st.expander(
                            f"How to receive with your {wallet_type_display}"
                        ):
                            st.write(wallet_specific_instructions)

                    # Show address and QR code for all wallet types with addresses
                    st.info("Send funds to this address:")
                    st.code(wallet.address)

                    # Generate QR code
                    st.write("QR Code:")
                    st.image(
                        f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={wallet.address}",
                        width=150,
                    )

                    # Add option to manually record a receive transaction
                    st.subheader("Record a received transaction")

                    with st.form(f"receive_form_{wallet.id}"):
                        currency = st.text_input("Currency")
                        amount = st.number_input(
                            "Amount Received", min_value=0.0, step=0.01
                        )
                        notes = st.text_area("Notes (optional)")

                        submitted = st.form_submit_button("Record")

                        if submitted:
                            if not currency:
                                st.error("Currency is required")
                            elif amount <= 0:
                                st.error("Amount must be greater than 0")
                            else:
                                if add_transaction(
                                    user_id=user_id,
                                    wallet_id=wallet.id,
                                    transaction_type="receive",
                                    currency=currency,
                                    amount=amount,
                                    notes=notes,
                                ):
                                    st.success(
                                        f"Recorded receipt of {amount} {currency}"
                                    )
                                    time.sleep(1)
                                    st.rerun()

        # Display balances
        st.subheader("Balances")

        balances = get_wallet_balances(wallet.id)

        if not balances:
            st.info("No balances found. Click 'Refresh' to update from exchange.")
        else:
            # Mock prices for demo
            if prices is None:
                prices = {
                    "BTC": 69420.0,
                    "ETH": 3500.0,
                    "ADA": 0.45,
                    "SOL": 95.0,
                    "DOGE": 0.12,
                    "USDT": 1.0,
                    "USDC": 1.0,
                }

            # Create a table of balances
            balance_data = []
            for balance in balances:
                price = prices.get(balance.currency, 0)
                usd_value = balance.amount * price

                balance_data.append(
                    {
                        "Currency": balance.currency,
                        "Amount": f"{balance.amount:.8f}".rstrip("0").rstrip("."),
                        "Price": f"${price:,.2f}" if price > 0 else "-",
                        "Value (USD)": f"${usd_value:,.2f}" if price > 0 else "-",
                    }
                )

            # Display as dataframe
            balance_df = pd.DataFrame(balance_data)
            st.dataframe(balance_df, hide_index=True, use_container_width=True)


def show_wallets():
    """Display the wallets page"""
    # Require login
    user = require_login()

    # Add page title
    st.subheader("Crypto Wallets")

    # Add wallet button
    show_add_wallet_form(user["id"])

    # Get wallets
    wallets = get_user_wallets(user["id"])

    if not wallets:
        st.info("You don't have any wallets yet. Add one to get started.")
    else:
        # Mock prices for demo
        prices = {
            "BTC": 69420.0,
            "ETH": 3500.0,
            "ADA": 0.45,
            "SOL": 95.0,
            "DOGE": 0.12,
            "USDT": 1.0,
            "USDC": 1.0,
        }

        # Display each wallet
        for wallet in wallets:
            show_wallet_card(wallet, user["id"], prices)
