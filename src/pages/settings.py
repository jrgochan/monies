import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os

from src.utils.auth import require_login, hash_password
from src.models.database import SessionLocal, User, ApiKey
from src.utils.security import store_api_key, get_api_key, encrypt_data, decrypt_data

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
    
    # Update email
    with st.form("update_email_form"):
        st.subheader("Update Email")
        
        new_email = st.text_input("New Email Address")
        
        submitted = st.form_submit_button("Update Email")
        
        if submitted:
            if not new_email:
                st.error("Please enter a new email address")
            else:
                success, message = update_user_email(user['id'], new_email)
                
                if success:
                    st.success(message)
                    # Update session state
                    st.session_state.user['email'] = new_email
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error(message)
    
    # Change password
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
                user_obj = authenticate_user(db, user['username'], current_password)
                db.close()
                
                if not user_obj:
                    st.error("Current password is incorrect")
                else:
                    # Update password
                    success, message = update_user_password(user['id'], new_password)
                    
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
            key_data.append({
                "ID": key.id,
                "Service": key.service,
                "Added": key.created_at.strftime("%Y-%m-%d")
            })
        
        key_df = pd.DataFrame(key_data)
        st.dataframe(key_df, hide_index=True, use_container_width=True)
        
        # Delete key option
        key_to_delete = st.selectbox(
            "Select API Key to Delete",
            options=[f"{k.service} (ID: {k.id})" for k in keys],
            index=None
        )
        
        if key_to_delete and st.button("Delete Selected API Key"):
            # Extract key ID from selection
            key_id = int(key_to_delete.split("ID: ")[1].split(")")[0])
            
            if delete_api_key(key_id):
                st.success("API key deleted successfully")
                time.sleep(1)
                st.experimental_rerun()
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
                    store_api_key(db, user_id, service, api_key, api_secret if api_secret else None)
                    st.success(f"API key for {service} added successfully")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error adding API key: {str(e)}")
                finally:
                    db.close()

def show_preferences(user_id):
    """Show app preferences section"""
    st.subheader("App Preferences")
    
    # Theme preference
    theme = st.radio(
        "Theme",
        options=["Light", "Dark", "System"],
        horizontal=True
    )
    
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
        if st.form_submit_button("Save Preferences"):
            st.success("Preferences saved successfully")
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
            "All Data"
        ]
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Format options
    format = st.radio(
        "Export Format",
        options=["CSV", "JSON", "Excel"],
        horizontal=True
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
                    mime="text/csv"
                )
            elif format == "JSON":
                content = '{"transactions": [{"date": "2023-01-01", "type": "buy", "currency": "BTC", "amount": 0.1}]}'
                st.download_button(
                    label="Download JSON",
                    data=content,
                    file_name=f"{export_type.lower().replace(' ', '_')}.json",
                    mime="application/json"
                )
            else:
                st.info("Excel export would be available in the full app")

def show_settings():
    """Display the Settings page"""
    # Require login
    user = require_login()
    
    # Add page title
    st.title("Settings")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Account", 
        "API Keys", 
        "Preferences",
        "Data Export"
    ])
    
    with tab1:
        show_account_settings(user)
    
    with tab2:
        show_api_key_settings(user['id'])
    
    with tab3:
        show_preferences(user['id'])
    
    with tab4:
        show_data_export(user['id'])