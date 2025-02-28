import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.models.database import ApiKey, DataSource, SessionLocal, User, UserDataSource
from src.utils.api_config import APIConfigManager
from src.utils.api_tester import APITester
from src.utils.auth import hash_password, require_login
from src.utils.data_aggregator import DataAggregator
from src.utils.security import decrypt_data, encrypt_data, get_api_key, store_api_key


def update_user_password(user_id, new_password):
    """Update a user's password"""
    db = SessionLocal()
    try:
        # Find user
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return False, "User not found"

        # Update password
        user.password_hash = hash_password(new_password)
        user.updated_at = datetime.utcnow()

        db.commit()
        return True, "Password updated successfully"
    except Exception as e:
        db.rollback()
        return False, f"Error updating password: {str(e)}"
    finally:
        db.close()


def update_user_email(user_id, new_email):
    """Update a user's email"""
    db = SessionLocal()
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == new_email).first()

        if existing and existing.id != user_id:
            return False, "Email already in use"

        # Find user
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return False, "User not found"

        # Update email
        user.email = new_email
        user.updated_at = datetime.utcnow()

        db.commit()
        return True, "Email updated successfully"
    except Exception as e:
        db.rollback()
        return False, f"Error updating email: {str(e)}"
    finally:
        db.close()


def get_user_api_keys(user_id):
    """Get all API keys for a user"""
    db = SessionLocal()
    try:
        keys = db.query(ApiKey).filter(ApiKey.user_id == user_id).all()
        return keys
    finally:
        db.close()


def delete_api_key(key_id):
    """Delete an API key"""
    db = SessionLocal()
    try:
        db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error deleting API key: {str(e)}")
        return False
    finally:
        db.close()


def show_account_settings(user):
    """Show account settings section"""
    st.subheader("Account Settings")

    # Display current account info
    st.write(f"Username: **{user['username']}**")
    st.write(f"Email: **{user['email']}**")
    st.write(f"Account created: **{user['created_at']}**")

    # Display OAuth provider info if connected
    if "oauth_provider" in user and user["oauth_provider"]:
        st.write(f"Connected with: **{user['oauth_provider'].capitalize()}**")
        st.info(
            f"Your account is linked to {user['oauth_provider'].capitalize()}. You can use {user['oauth_provider'].capitalize()} to log in."
        )

    # Connected accounts section
    st.subheader("Connected Accounts")

    # Get the database session to check current connections
    db = SessionLocal()
    user_obj = db.query(User).filter(User.id == user["id"]).first()
    db.close()

    # Import OAuth config here to avoid circular imports
    from src.utils.oauth_config import OAUTH_CONFIGS, generate_oauth_authorize_url

    # Get available and configured OAuth providers
    available_providers = []
    for provider, config in OAUTH_CONFIGS.items():
        if config["client_id"] and config["client_secret"]:
            available_providers.append(provider)

    if not available_providers:
        st.info(
            "No OAuth providers are configured. Please set up OAuth credentials in your environment variables."
        )
    else:
        # Create a grid of provider cards, 3 per row
        providers_per_row = 3
        rows = (len(available_providers) + providers_per_row - 1) // providers_per_row

        for row in range(rows):
            cols = st.columns(providers_per_row)

            for i in range(providers_per_row):
                idx = row * providers_per_row + i
                if idx < len(available_providers):
                    provider = available_providers[idx]
                    config = OAUTH_CONFIGS[provider]

                    # Check if connected to this provider
                    is_connected = (
                        user_obj.oauth_provider == provider if user_obj else False
                    )

                    with cols[i]:
                        with st.container(border=True):
                            st.markdown(f"### {config['display_name']}")

                            # Show status and action button
                            if is_connected:
                                st.success("Connected ✓")

                                # Add disconnect button
                                if st.button(
                                    f"Disconnect {config['display_name']}",
                                    key=f"disconnect_{provider}",
                                ):
                                    # This would clear the OAuth connection, but we'll just show a message for now
                                    st.warning(
                                        f"This would disconnect your {config['display_name']} account. This functionality is not implemented yet."
                                    )
                            else:
                                st.info("Not connected")

                                # Add connect button
                                if st.button(
                                    f"Connect {config['display_name']}",
                                    key=f"connect_{provider}",
                                ):
                                    # Generate OAuth URL and store state in session
                                    auth_url, state = generate_oauth_authorize_url(
                                        provider
                                    )
                                    if auth_url:
                                        # Store state in session
                                        st.session_state.oauth_state = state
                                        st.session_state.oauth_flow = provider

                                        # Redirect to OAuth provider
                                        st.markdown(
                                            f'<meta http-equiv="refresh" content="0;url={auth_url}">',
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.error(
                                            f"Failed to generate {config['display_name']} OAuth URL. Please check your configuration."
                                        )

    st.markdown("---")

    # Update email
    with st.form("update_email_form"):
        st.subheader("Update Email")

        new_email = st.text_input("New Email Address")

        submitted = st.form_submit_button("Update Email")

        if submitted:
            if not new_email:
                st.error("Please enter a new email address")
            else:
                success, message = update_user_email(user["id"], new_email)

                if success:
                    st.success(message)
                    # Update session state
                    st.session_state.user["email"] = new_email
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)

    # Only show password change form if not using OAuth or has a password already
    if not user.get("oauth_provider") or user_obj.password_hash:
        with st.form("change_password_form"):
            st.subheader("Change Password")

            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            submitted = st.form_submit_button("Change Password")

            if submitted:
                if not current_password or not new_password or not confirm_password:
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    # Verify current password
                    from src.utils.auth import authenticate_user

                    db = SessionLocal()
                    user_obj = authenticate_user(db, user["username"], current_password)
                    db.close()

                    if not user_obj:
                        st.error("Current password is incorrect")
                    else:
                        # Update password
                        success, message = update_user_password(
                            user["id"], new_password
                        )

                        if success:
                            st.success(message)
                        else:
                            st.error(message)
    # If user is using OAuth and doesn't have a password, show an option to set up a password
    elif user.get("oauth_provider") and not user_obj.password_hash:
        with st.form("setup_password_form"):
            st.subheader("Set Up Password")
            st.info(
                f"You're currently using {user.get('oauth_provider').capitalize()} to log in. Setting up a password will allow you to log in with your username and password as well."
            )

            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            submitted = st.form_submit_button("Set Password")

            if submitted:
                if not new_password or not confirm_password:
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    # Set the password
                    success, message = update_user_password(user["id"], new_password)

                    if success:
                        st.success(message)
                    else:
                        st.error(message)


def show_api_key_settings(user_id):
    """Show API key management section"""
    st.subheader("API Key Management")

    # Get user's API keys
    keys = get_user_api_keys(user_id)

    # Display keys
    if keys:
        key_data = []
        for key in keys:
            key_data.append(
                {
                    "ID": key.id,
                    "Service": key.service,
                    "Added": key.created_at.strftime("%Y-%m-%d"),
                }
            )

        key_df = pd.DataFrame(key_data)
        st.dataframe(key_df, hide_index=True, use_container_width=True)

        # Delete key option
        key_to_delete = st.selectbox(
            "Select API Key to Delete",
            options=[f"{k.service} (ID: {k.id})" for k in keys],
            index=None,
        )

        if key_to_delete and st.button("Delete Selected API Key"):
            # Extract key ID from selection
            key_id = int(key_to_delete.split("ID: ")[1].split(")")[0])

            if delete_api_key(key_id):
                st.success("API key deleted successfully")
                time.sleep(1)
                st.rerun()
    else:
        st.info("No API keys found.")

    # Add new API key
    with st.form("add_api_key_form"):
        st.subheader("Add New API Key")

        service = st.text_input("Service Name", placeholder="binance, openai, etc.")
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret (optional)", type="password")

        submitted = st.form_submit_button("Add API Key")

        if submitted:
            if not service or not api_key:
                st.error("Service name and API key are required")
            else:
                # Add API key
                db = SessionLocal()
                try:
                    store_api_key(
                        db,
                        user_id,
                        service,
                        api_key,
                        api_secret if api_secret else None,
                    )
                    st.success(f"API key for {service} added successfully")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding API key: {str(e)}")
                finally:
                    db.close()


def show_preferences(user_id):
    """Show app preferences section"""
    st.subheader("App Preferences")

    # Theme preference
    theme = st.radio("Theme", options=["Light", "Dark", "System"], horizontal=True)

    # Data source preferences section
    st.subheader("Data Source Preferences")
    st.info(
        "Configure which data sources to use for financial data and their priority."
    )

    # Get the database session
    db = SessionLocal()

    try:
        # Get all available data sources
        all_categories = ["stocks", "crypto", "etf"]

        # Create tabs for each category
        tabs = st.tabs(["Stocks", "Cryptocurrencies", "ETFs"])

        for i, category in enumerate(all_categories):
            with tabs[i]:
                # Get data sources for this category
                data_sources = DataAggregator.get_user_data_sources(
                    db, user_id, category
                )

                if not data_sources:
                    st.warning(f"No data sources available for {category}.")
                    continue

                st.markdown(f"### {category.title()} Data Sources")
                st.markdown(
                    "Drag to reorder sources by priority. Sources higher in the list will be tried first."
                )

                # Create a form for each category
                with st.form(f"data_source_form_{category}"):
                    # Create a list of enabled sources with checkboxes
                    enabled_sources = []
                    for ds in data_sources:
                        enabled = st.checkbox(
                            f"{ds['display_name']} ({ds['name']})",
                            value=ds["enabled"],
                            key=f"ds_{ds['id']}_{category}",
                        )

                        # Show API requirement info
                        if ds["api_required"]:
                            if ds["api_key_field"]:
                                env_value = os.environ.get(ds["api_key_field"], "")
                                if env_value:
                                    st.info(
                                        f"API key found in environment variable {ds['api_key_field']}"
                                    )
                                else:
                                    st.warning(
                                        f"API key required in environment variable {ds['api_key_field']}"
                                    )

                        enabled_sources.append(
                            {"data_source_id": ds["id"], "enabled": enabled}
                        )

                    # Aggregation options
                    st.markdown("### Aggregation Options")

                    aggregate_data = st.checkbox(
                        "Aggregate data from multiple sources",
                        value=True,
                        help="When enabled, data from multiple sources will be combined and averaged.",
                    )

                    min_sources = st.slider(
                        "Minimum sources for aggregation",
                        min_value=1,
                        max_value=len(data_sources),
                        value=min(2, len(data_sources)),
                        help="Minimum number of data sources required for aggregation.",
                    )

                    # Get user's current preferences
                    user_obj = db.query(User).filter(User.id == user_id).first()
                    preferences = {}

                    if user_obj and user_obj.data_source_preferences:
                        try:
                            preferences = json.loads(user_obj.data_source_preferences)
                        except:
                            preferences = {}

                    # Update preferences
                    if category not in preferences:
                        preferences[category] = {}

                    preferences[category]["aggregate"] = aggregate_data
                    preferences[category]["min_sources"] = min_sources

                    # Save button
                    if st.form_submit_button(f"Save {category.title()} Preferences"):
                        # Update enabled/disabled state
                        DataAggregator.update_user_data_source_preferences(
                            db, user_id, enabled_sources
                        )

                        # Update user preferences
                        user_obj = db.query(User).filter(User.id == user_id).first()
                        if user_obj:
                            user_obj.data_source_preferences = json.dumps(preferences)
                            db.commit()

                        st.success(
                            f"{category.title()} data source preferences saved successfully!"
                        )

    finally:
        db.close()

    # Notification settings
    with st.form("notification_settings"):
        st.subheader("Notification Settings")

        email_notifications = st.toggle("Email Notifications")

        if email_notifications:
            st.checkbox("Price alerts", value=True)
            st.checkbox("Trading confirmations", value=True)
            st.checkbox("Social post notifications", value=False)
            st.checkbox("AI insights updates", value=True)

        # Save button
        if st.form_submit_button("Save Notification Preferences"):
            st.success("Notification preferences saved successfully")
            # In a real app, save to database


def show_data_export(user_id):
    """Show data export options"""
    st.subheader("Data Export")

    # Export options
    export_type = st.selectbox(
        "Select Data to Export",
        options=[
            "Transaction History",
            "Wallet Balances",
            "Portfolio Performance",
            "All Data",
        ],
    )

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")

    # Format options
    format = st.radio(
        "Export Format", options=["CSV", "JSON", "Excel"], horizontal=True
    )

    # Export button
    if st.button("Export Data"):
        with st.spinner("Preparing export..."):
            # In a real app, query the database and format data
            # For demo, just show a success message
            time.sleep(2)
            st.success(f"{export_type} exported successfully as {format}")

            # Create download button with dummy data
            if format == "CSV":
                content = "Date,Type,Currency,Amount\n2023-01-01,buy,BTC,0.1\n2023-01-02,sell,ETH,2.5"
                st.download_button(
                    label="Download CSV",
                    data=content,
                    file_name=f"{export_type.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            elif format == "JSON":
                content = '{"transactions": [{"date": "2023-01-01", "type": "buy", "currency": "BTC", "amount": 0.1}]}'
                st.download_button(
                    label="Download JSON",
                    data=content,
                    file_name=f"{export_type.lower().replace(' ', '_')}.json",
                    mime="application/json",
                )
            else:
                st.info("Excel export would be available in the full app")


def test_all_connections():
    """Test all API connections and return results"""
    results = []

    for api in APIConfigManager.get_api_configs():
        # Check if we have the necessary credentials
        key, secret = APIConfigManager.get_api_credentials(api["service_id"])
        has_credentials = True

        if api.get("needs_key", True) and not key:
            has_credentials = False

        if api.get("needs_secret", False) and not secret:
            has_credentials = False

        # Only test if we have credentials or no credentials are needed
        if has_credentials or not api.get("needs_key", True):
            try:
                success, message = APIConfigManager.test_api_connection(
                    api["service_id"]
                )
            except Exception as e:
                success = False
                message = f"Error during test: {str(e)}"
        else:
            success = False
            message = "Missing credentials"

        results.append(
            {
                "name": api["name"],
                "service_id": api["service_id"],
                "category": api.get("category", "Other"),
                "has_credentials": has_credentials,
                "success": success,
                "message": message,
                "api_config": api,
            }
        )

    return results


def show_connection_status():
    """Show API connection status dashboard"""
    st.subheader("API Connection Status")

    # Add a refresh button
    if st.button("Refresh All Connections"):
        with st.spinner("Testing all API connections..."):
            st.session_state.connection_results = test_all_connections()

    # Initialize results if not in session state
    if "connection_results" not in st.session_state:
        with st.spinner("Testing all API connections..."):
            st.session_state.connection_results = test_all_connections()

    # Get results
    results = st.session_state.connection_results

    # Create a card for each category
    categories = APIConfigManager.get_api_categories()

    # Overall stats
    total_apis = len(results)
    configured_apis = sum(
        1
        for r in results
        if r["has_credentials"] or not r["api_config"].get("needs_key", True)
    )
    working_apis = sum(1 for r in results if r["success"])

    st.markdown(f"### Connection Summary")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total APIs", total_apis)
    with cols[1]:
        st.metric("Configured APIs", configured_apis)
    with cols[2]:
        st.metric("Working Connections", working_apis)

    st.markdown("---")

    # Display status for each category
    for category in categories:
        # Get results for this category
        category_results = [r for r in results if r["category"] == category]

        if not category_results:
            continue

        st.markdown(f"### {category} APIs")

        # Create a grid of cards
        cols = st.columns(3)
        col_index = 0

        for result in category_results:
            with cols[col_index]:
                # Determine status color
                if result["success"]:
                    status_color = "green"
                    status_emoji = "✅"
                elif not result["has_credentials"] and result["api_config"].get(
                    "needs_key", True
                ):
                    status_color = "gray"
                    status_emoji = "⚪"
                else:
                    status_color = "red"
                    status_emoji = "❌"

                # Create card
                with st.container(border=True):
                    st.markdown(f"#### {status_emoji} {result['name']}")

                    # Status message
                    if result["success"]:
                        st.markdown(
                            f"**Status:** <span style='color:{status_color}'>Connected</span>",
                            unsafe_allow_html=True,
                        )
                    elif not result["has_credentials"] and result["api_config"].get(
                        "needs_key", True
                    ):
                        st.markdown(
                            f"**Status:** <span style='color:{status_color}'>Not Configured</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"**Status:** <span style='color:{status_color}'>Connection Error</span>",
                            unsafe_allow_html=True,
                        )

                    # Show message on hover with tooltip
                    if len(result["message"]) > 50:
                        short_message = result["message"][:50] + "..."
                    else:
                        short_message = result["message"]

                    st.markdown(f"**Message:** {short_message}")

                    # Add quick test button
                    if st.button("Test", key=f"quick_test_{result['service_id']}"):
                        with st.spinner(f"Testing connection to {result['name']}..."):
                            success, message = APIConfigManager.test_api_connection(
                                result["service_id"]
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)

                    # Add configure link that jumps to the Configuration tab and specific API section
                    if st.button("Configure", key=f"config_{result['service_id']}"):
                        # Set session state to remember which API to expand
                        st.session_state.selected_api_to_configure = result[
                            "service_id"
                        ]
                        # Switch to the configuration tab
                        st.session_state.settings_active_tab = 1
                        # Use rerun to apply changes
                        st.rerun()

            # Move to next column
            col_index = (col_index + 1) % 3

        st.markdown("---")

    # Add a refresh button at the bottom too
    if st.button("Refresh All Connections", key="refresh_bottom"):
        with st.spinner("Testing all API connections..."):
            st.session_state.connection_results = test_all_connections()


def show_api_troubleshooting():
    """Show API troubleshooting guidance"""
    st.subheader("API Connection Troubleshooting")

    st.markdown(
        """
    ### Common Connection Issues

    If you're experiencing connection issues with any API, here are some common problems and solutions:

    #### Network Issues
    - **Firewall Blocking**: Your firewall may be blocking outgoing connections to API endpoints
    - **Proxy Configuration**: If you're behind a corporate proxy, you may need to configure it
    - **VPN Interference**: VPNs can sometimes interfere with API connections
    - **DNS Issues**: DNS resolution problems can prevent connecting to API endpoints

    #### API Key Issues
    - **Invalid API Key**: Ensure your API key is correctly copied without extra spaces
    - **Expired API Key**: Some services have keys that expire and need to be refreshed
    - **Rate Limiting**: Many APIs have usage limits that may be exceeded
    - **IP Restrictions**: Some API keys are restricted to specific IP addresses

    #### Service-Specific Issues
    """
    )

    # Create expanders for specific service troubleshooting
    with st.expander("Yahoo Finance (yfinance) Issues"):
        st.markdown(
            """
        #### Yahoo Finance Connection Issues

        Yahoo Finance doesn't require API keys but can still have connection issues:

        - **Rate Limiting**: Yahoo Finance may rate-limit requests if too many are made
        - **Blocked IPs**: Some IPs may be temporarily blocked due to excessive requests
        - **Symbol Format**: Make sure you're using the correct ticker symbol format
        - **Proxy Required**: Some regions may require a proxy to access Yahoo Finance

        **Solution**: Try the following:
        1. Add a small delay between requests (already implemented in the app)
        2. Use a VPN if your location is restricted
        3. Try different ticker formats (e.g., BTC-USD, BTCUSD=X)
        4. Verify the symbol exists and is active
        """
        )

    with st.expander("Crypto Exchange Issues"):
        st.markdown(
            """
        #### Crypto Exchange Connection Issues

        Common issues with crypto exchange APIs:

        - **Geographical Restrictions**: Some exchanges restrict access from certain countries
        - **KYC Requirements**: API may require additional verification
        - **Wrong Endpoint**: Using regular Binance instead of Binance.US in restricted regions
        - **Permission Settings**: API keys may not have the correct permissions enabled

        **Solution**: Try the following:
        1. For Binance users in the US, make sure to use Binance.US API
        2. Check that your API key has the correct permissions (read-only may be enough)
        3. Create a new API key if you suspect the old one is compromised or expired
        4. Use a VPN if your region is restricted (but ensure this doesn't violate terms of service)

        #### Coinbase-Specific Issues

        - **API Key Format**: Make sure you're using an API key from Coinbase (not Coinbase Pro/Advanced Trade)
        - **CDP API Keys**: We support Coinbase Developer Platform API keys that include organization IDs (format: organization_id:api_key)
        - **API Secret Format**: The API secret should be Base64 encoded
        - **Permissions**: Ensure your API key has the correct permissions (wallet:accounts:read at minimum)
        - **2FA Requirement**: Coinbase may require additional verification for certain actions

        **To create a Coinbase API key**:
        1. Log in to your Coinbase account
        2. Go to Settings > API > New API Key
        3. Enable the required permissions (wallet:accounts:read, wallet:transactions:read)
        4. Complete 2FA verification if prompted
        5. Copy both the API Key and API Secret

        **To create a Coinbase Developer Platform (CDP) API key**:
        1. Log in to your Coinbase Developer account at developer.coinbase.com
        2. Go to My Apps > Create New API Key
        3. Select the permissions required for your application
        4. Note the API key format with organization ID (organization_id:api_key)
        5. You can use this key directly - our app will handle the organization ID automatically
        """
        )

    with st.expander("AI/LLM API Issues"):
        st.markdown(
            """
        #### AI/LLM API Connection Issues

        Common issues with AI API connections:

        - **API Key Format**: Some services have complex API key formats
        - **Organization ID**: Some services require an organization ID in addition to the API key
        - **Model Availability**: The requested model may not be available or may be deprecated
        - **Usage Limits**: You may have exceeded your usage quota

        **Solution**: Try the following:
        1. Check your current usage on the provider's dashboard
        2. Ensure the requested model is available in your subscription tier
        3. For OpenAI, check if you need to migrate to a newer API version
        4. For locally hosted models like Ollama, ensure the service is running
        """
        )

    st.markdown(
        """
    ### Diagnostic Tests

    You can run network diagnostic tests to help identify connectivity issues:
    """
    )

    # Network diagnostic tools
    with st.expander("Network Diagnostics"):
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("Run Connection Test"):
                with st.spinner("Testing internet connection..."):
                    try:
                        import socket

                        # Test basic internet connectivity
                        socket.create_connection(("8.8.8.8", 53), timeout=5)
                        st.success("✅ Basic internet connectivity: Good")
                    except Exception as e:
                        st.error(f"❌ Basic internet connectivity: Failed ({str(e)})")

                    # Test API endpoint connectivity
                    endpoints = [
                        ("api.openai.com", 443, "OpenAI API"),
                        ("api.binance.us", 443, "Binance.US API"),
                        ("api.binance.com", 443, "Binance API"),
                        ("api.coinbase.com", 443, "Coinbase API"),
                        ("query1.finance.yahoo.com", 443, "Yahoo Finance API"),
                        ("api.coingecko.com", 443, "CoinGecko API"),
                    ]

                    for host, port, name in endpoints:
                        try:
                            socket.create_connection((host, port), timeout=5)
                            st.success(f"✅ {name} endpoint ({host}): Reachable")
                        except Exception as e:
                            st.error(
                                f"❌ {name} endpoint ({host}): Unreachable ({str(e)})"
                            )

        with col2:
            st.markdown(
                """
            This test checks basic internet connectivity and the reachability of various API endpoints.

            - A successful test shows that your network can reach the API endpoints
            - A failed test indicates potential network or firewall issues

            If endpoints are unreachable, you may need to:
            1. Check your firewall settings
            2. Configure your proxy (if applicable)
            3. Try a different network
            4. Use a VPN (where allowed by terms of service)
            """
            )

    st.markdown(
        """
    ### Updating Configuration

    For security reasons, API credentials are stored in two places:

    1. Your **.env** file in the project root directory
    2. The **application database** (encrypted)

    When updating API credentials, both locations are updated automatically. If you need to manually edit the .env file, make sure to restart the application afterward.
    """
    )


def show_api_connections(user_id):
    """Show API connection configuration and testing page"""
    st.subheader("API Connections")

    # Add tabs for status dashboard and configuration
    status_tab, config_tab, troubleshoot_tab = st.tabs(
        ["Connection Status", "Configuration", "Troubleshooting"]
    )

    with status_tab:
        show_connection_status()

    with config_tab:
        # Get API categories
        categories = APIConfigManager.get_api_categories()

        # If we have a selected service ID, find which category tab it belongs to
        selected_category_index = 0
        if "selected_api_to_configure" in st.session_state:
            selected_service = st.session_state.selected_api_to_configure
            for i, category in enumerate(categories):
                apis = APIConfigManager.get_api_configs_by_category(category)
                if any(api["service_id"] == selected_service for api in apis):
                    selected_category_index = i
                    break

        # Create category tabs
        tabs = st.tabs(categories)

        for i, category in enumerate(categories):
            with tabs[i]:
                # Get APIs in this category
                apis = APIConfigManager.get_api_configs_by_category(category)

                st.markdown(f"### {category} API Configuration")
                st.markdown(
                    "Configure and test connections to various APIs in this category."
                )

                # Create an expansion section for each API
                for api in apis:
                    # Check if we should auto-expand this API's configuration
                    auto_expand = (
                        "selected_api_to_configure" in st.session_state
                        and st.session_state.selected_api_to_configure
                        == api["service_id"]
                    )

                    with st.expander(
                        f"{api['name']} Configuration", expanded=auto_expand
                    ):
                        # Display API info
                        st.markdown(f"**{api['name']}** (ID: {api['service_id']})")
                        st.markdown(f"{api['description']}")
                        st.markdown(
                            f"[Website]({api['website']}) | [API Documentation]({api['api_docs']})"
                        )

                        # Get current values
                        current_key = ""
                        current_secret = ""

                        if api.get("env_var_key"):
                            current_key = APIConfigManager.get_api_value_from_env(
                                api.get("env_var_key")
                            )

                        if api.get("env_var_secret"):
                            current_secret = APIConfigManager.get_api_value_from_env(
                                api.get("env_var_secret")
                            )

                        # Display current status and test button (outside the form)
                        if not api.get("needs_key", True) or (
                            current_key
                            and (not api.get("needs_secret", False) or current_secret)
                        ):
                            # We have enough info to test the connection
                            st.markdown("**Current Status**")
                            if st.button(
                                f"Test Connection", key=f"test_{api['service_id']}"
                            ):
                                with st.spinner(
                                    f"Testing connection to {api['name']}..."
                                ):
                                    (
                                        success,
                                        message,
                                    ) = APIConfigManager.test_api_connection(
                                        api["service_id"]
                                    )
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                            st.markdown("---")

                        # Create configuration form
                        with st.form(f"{api['service_id']}_config_form"):
                            # Special handling for URL-based services like Ollama
                            if api.get("is_url", False):
                                key_label = "Server URL"
                                key_help = f"URL of the {api['name']} server (e.g., {api.get('default_url', '')})"
                                secret_required = False
                            else:
                                key_label = (
                                    "API Key"
                                    if api.get("needs_key", True)
                                    else "Not Required"
                                )
                                key_help = (
                                    f"Your {api['name']} API key"
                                    if api.get("needs_key", True)
                                    else "No API key required for this service"
                                )
                                secret_required = api.get("needs_secret", False)

                            # API key input
                            if api.get("needs_key", True):
                                api_key = st.text_input(
                                    key_label,
                                    value=current_key if current_key else "",
                                    type="password" if current_key else "default",
                                    help=key_help,
                                )
                            else:
                                st.info("This API doesn't require an API key")
                                api_key = None

                            # API secret input (if needed)
                            if secret_required:
                                api_secret = st.text_input(
                                    "API Secret",
                                    value=current_secret if current_secret else "",
                                    type="password" if current_secret else "default",
                                    help=f"Your {api['name']} API secret",
                                )
                            else:
                                api_secret = None

                            # Environment variable names
                            st.markdown("**Environment Variables**")
                            if api.get("env_var_key"):
                                st.code(api.get("env_var_key"))
                            if api.get("env_var_secret"):
                                st.code(api.get("env_var_secret"))

                            # Submit button
                            submitted = st.form_submit_button("Save Configuration")

                            if submitted:
                                # Update API configuration
                                if api.get("needs_key", True) and not api_key:
                                    st.error(f"API key is required for {api['name']}")
                                elif secret_required and not api_secret:
                                    st.error(
                                        f"API secret is required for {api['name']}"
                                    )
                                else:
                                    db = SessionLocal()
                                    try:
                                        (
                                            success,
                                            message,
                                        ) = APIConfigManager.save_api_credentials(
                                            api["service_id"],
                                            api_key,
                                            api_secret,
                                            db,
                                            user_id,
                                        )

                                        if success:
                                            st.success(message)

                                            # Test the connection after saving
                                            with st.spinner(
                                                f"Testing connection to {api['name']}..."
                                            ):
                                                (
                                                    test_success,
                                                    test_message,
                                                ) = APIConfigManager.test_api_connection(
                                                    api["service_id"]
                                                )
                                                if test_success:
                                                    st.success(test_message)

                                                    # Update session state with new test results
                                                    if (
                                                        "connection_results"
                                                        in st.session_state
                                                    ):
                                                        for (
                                                            result
                                                        ) in (
                                                            st.session_state.connection_results
                                                        ):
                                                            if (
                                                                result["service_id"]
                                                                == api["service_id"]
                                                            ):
                                                                result["success"] = True
                                                                result[
                                                                    "message"
                                                                ] = test_message
                                                                result[
                                                                    "has_credentials"
                                                                ] = True
                                                else:
                                                    st.warning(
                                                        f"Saved credentials but connection test failed: {test_message}"
                                                    )

                                                    # Update session state with new test results
                                                    if (
                                                        "connection_results"
                                                        in st.session_state
                                                    ):
                                                        for (
                                                            result
                                                        ) in (
                                                            st.session_state.connection_results
                                                        ):
                                                            if (
                                                                result["service_id"]
                                                                == api["service_id"]
                                                            ):
                                                                result[
                                                                    "success"
                                                                ] = False
                                                                result[
                                                                    "message"
                                                                ] = test_message
                                                                result[
                                                                    "has_credentials"
                                                                ] = True
                                        else:
                                            st.error(message)
                                    finally:
                                        db.close()

                        # Add a divider between APIs
                        st.markdown("---")

    with troubleshoot_tab:
        show_api_troubleshooting()


def show_settings():
    """Display the Settings page"""
    # Require login
    user = require_login()

    # Add page title
    st.title("Settings")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Account", "API Keys", "API Connections", "Preferences", "Data Export"]
    )

    with tab1:
        show_account_settings(user)

    with tab2:
        show_api_key_settings(user["id"])

    with tab3:
        show_api_connections(user["id"])

    with tab4:
        show_preferences(user["id"])

    with tab5:
        show_data_export(user["id"])
