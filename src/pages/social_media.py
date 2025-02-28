"""Social media integration page for connecting and posting to social media platforms.

This module handles:
- Connecting social media accounts
- Creating and scheduling posts
- Managing scheduled posts
- Analyzing social media sentiment
"""

import base64
import os
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.api.ai_analysis import generate_social_post
from src.api.social_media import (
    cancel_scheduled_post,
    get_scheduled_posts,
    get_twitter_sentiment,
    post_to_facebook,
    post_to_linkedin,
    post_to_twitter,
    schedule_post,
)
from src.models.database import ScheduledPost, SessionLocal, SocialAccount
from src.utils.auth import require_login
from src.utils.security import get_api_key, store_api_key

# Rename this module to avoid conflicts with src.api.social_media
# This file should be imported as src.pages.social_media_page


def get_user_social_accounts(user_id):
    """Get all social media accounts for a user."""
    db = SessionLocal()
    try:
        accounts = (
            db.query(SocialAccount).filter(SocialAccount.user_id == user_id).all()
        )
        return accounts
    finally:
        db.close()


def add_social_account(user_id, platform, username, token, token_secret=None):
    """Add a new social media account for a user."""
    db = SessionLocal()
    try:
        # Check if account already exists
        existing = (
            db.query(SocialAccount)
            .filter(
                SocialAccount.user_id == user_id, SocialAccount.platform == platform
            )
            .first()
        )

        if existing:
            # Update existing
            existing.username = username
            existing.encrypted_token = token
            if token_secret is not None:
                existing.encrypted_token_secret = token_secret
            db.commit()
            return existing
        else:
            # Create new
            account = SocialAccount(
                user_id=user_id,
                platform=platform,
                username=username,
                encrypted_token=token,
                encrypted_token_secret=token_secret,
            )
            db.add(account)
            db.commit()
            db.refresh(account)
            return account
    except Exception as e:
        db.rollback()
        st.error(f"Error adding social account: {str(e)}")
        return None
    finally:
        db.close()


def delete_social_account(account_id):
    """Delete a social media account."""
    db = SessionLocal()
    try:
        db.query(SocialAccount).filter(SocialAccount.id == account_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        st.error(f"Error deleting social account: {str(e)}")
        return False
    finally:
        db.close()


def show_accounts_section(user_id):
    """Show social media accounts section."""
    st.subheader("Connected Accounts")

    # Get user's social accounts
    accounts = get_user_social_accounts(user_id)

    # Display accounts
    if accounts:
        for account in accounts:
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    if account.platform == "twitter":
                        st.markdown("🐦 **Twitter**")
                    elif account.platform == "facebook":
                        st.markdown("📘 **Facebook**")
                    elif account.platform == "linkedin":
                        st.markdown("🔵 **LinkedIn**")
                    elif account.platform == "instagram":
                        st.markdown("📸 **Instagram**")

                with col2:
                    st.markdown(f"**@{account.username}**")
                    st.caption(
                        f"Connected since {account.connected_at.strftime('%Y-%m-%d')}"
                    )

                with col3:
                    if st.button("Disconnect", key=f"disconnect_{account.id}"):
                        if delete_social_account(account.id):
                            st.success(f"Disconnected {account.platform} account")
                            time.sleep(1)
                            st.rerun()
    else:
        st.info("No social media accounts connected yet.")

    # Add account form
    with st.expander("Connect New Account", expanded=False):
        with st.form("add_social_form"):
            st.write("Connect a social media account")

            platform = st.selectbox(
                "Platform",
                options=["twitter", "facebook", "linkedin", "instagram"],
                format_func=lambda x: {
                    "twitter": "Twitter",
                    "facebook": "Facebook",
                    "linkedin": "LinkedIn",
                    "instagram": "Instagram",
                }.get(x, x),
            )

            username = st.text_input("Username")

            # Different fields based on platform
            if platform == "twitter":
                api_key = st.text_input("API Key (Consumer Key)", type="password")
                api_secret = st.text_input(
                    "API Secret (Consumer Secret)", type="password"
                )
                access_token = st.text_input("Access Token", type="password")
                access_secret = st.text_input("Access Token Secret", type="password")

                token = access_token
                token_secret = access_secret

                submitted = st.form_submit_button("Connect Account")

                if submitted:
                    if (
                        not username
                        or not api_key
                        or not api_secret
                        or not access_token
                        or not access_secret
                    ):
                        st.error("All fields are required")
                    else:
                        # Store API keys
                        db = SessionLocal()
                        store_api_key(db, user_id, "twitter_api", api_key, api_secret)
                        db.close()

                        # Add account
                        if add_social_account(
                            user_id, platform, username, token, token_secret
                        ):
                            st.success(f"Connected Twitter account @{username}")
                            time.sleep(1)
                            st.rerun()

            elif platform == "facebook":
                access_token = st.text_input("Page Access Token", type="password")

                token = access_token
                token_secret = None

                submitted = st.form_submit_button("Connect Account")

                if submitted:
                    if not username or not access_token:
                        st.error("All fields are required")
                    else:
                        # Add account
                        if add_social_account(
                            user_id, platform, username, token, token_secret
                        ):
                            st.success(f"Connected Facebook account {username}")
                            time.sleep(1)
                            st.rerun()

            elif platform == "linkedin":
                access_token = st.text_input("Access Token", type="password")

                token = access_token
                token_secret = None

                submitted = st.form_submit_button("Connect Account")

                if submitted:
                    if not username or not access_token:
                        st.error("All fields are required")
                    else:
                        # Add account
                        if add_social_account(
                            user_id, platform, username, token, token_secret
                        ):
                            st.success(f"Connected LinkedIn account {username}")
                            time.sleep(1)
                            st.rerun()

            else:  # Instagram
                access_token = st.text_input("Access Token", type="password")

                token = access_token
                token_secret = None

                submitted = st.form_submit_button("Connect Account")

                if submitted:
                    if not username or not access_token:
                        st.error("All fields are required")
                    else:
                        # Add account
                        if add_social_account(
                            user_id, platform, username, token, token_secret
                        ):
                            st.success(f"Connected Instagram account {username}")
                            time.sleep(1)
                            st.rerun()


def show_post_section(user_id):
    """Show post creation and scheduling section."""
    st.subheader("Create Post")

    # Get user's social accounts
    accounts = get_user_social_accounts(user_id)

    if not accounts:
        st.warning(
            "You need to connect at least one social media account to post. Use the 'Connected Accounts' section above."
        )
        return

    # Create tabs for manual and AI-generated posts
    post_tab1, post_tab2 = st.tabs(["Write Post", "AI-Generated Post"])

    with post_tab1:
        # Manual post creation
        with st.form("manual_post_form"):
            # Select platforms to post to
            platform_options = {
                a.platform: f"{a.platform.capitalize()} (@{a.username})"
                for a in accounts
            }
            selected_platforms = st.multiselect(
                "Post to",
                options=list(platform_options.keys()),
                format_func=lambda x: platform_options[x],
            )

            # Post content
            content = st.text_area(
                "Post Content",
                placeholder="Share your investment insights...",
                max_chars=280,  # Twitter limit
            )

            # File upload for media
            media_file = st.file_uploader(
                "Attach Media (optional)", type=["jpg", "jpeg", "png"]
            )

            # Scheduling options
            schedule_post = st.checkbox("Schedule for later")

            scheduled_time = None
            if schedule_post:
                min_date = datetime.now() + timedelta(minutes=5)
                scheduled_date = st.date_input("Date", min_value=min_date.date())
                scheduled_time_input = st.time_input("Time", value=min_date.time())

                scheduled_time = datetime.combine(scheduled_date, scheduled_time_input)

                if scheduled_time < datetime.now():
                    st.error("Scheduled time must be in the future")

            # Submit button
            submitted = st.form_submit_button(
                "Post" if not schedule_post else "Schedule"
            )

            if submitted:
                if not selected_platforms:
                    st.error("Please select at least one platform to post to")
                elif not content:
                    st.error("Post content is required")
                elif schedule_post and scheduled_time < datetime.now():
                    st.error("Scheduled time must be in the future")
                else:
                    # Save media file if provided
                    media_path = None
                    if media_file:
                        # Save the uploaded file to a temporary location
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(media_file.name)[1]
                        ) as tmp:
                            tmp.write(media_file.getvalue())
                            media_path = tmp.name

                    # Post or schedule
                    if schedule_post:
                        # Schedule the post
                        for platform in selected_platforms:
                            result = schedule_post(
                                user_id=user_id,
                                platform=platform,
                                content=content,
                                scheduled_time=scheduled_time,
                                media_path=media_path,
                            )

                            if result["success"]:
                                st.success(
                                    f"Post scheduled for {platform.capitalize()} at {scheduled_time.strftime('%Y-%m-%d %H:%M')}"
                                )
                            else:
                                st.error(
                                    f"Error scheduling post for {platform.capitalize()}: {result['message']}"
                                )
                    else:
                        # Post immediately
                        for platform in selected_platforms:
                            account = next(
                                (a for a in accounts if a.platform == platform), None
                            )

                            if not account:
                                st.error(f"Account for {platform} not found")
                                continue

                            with st.spinner(f"Posting to {platform.capitalize()}..."):
                                try:
                                    # Different API call based on platform
                                    if platform == "twitter":
                                        # Get API keys
                                        db = SessionLocal()
                                        api_key, api_secret = get_api_key(
                                            db, user_id, "twitter_api"
                                        )
                                        db.close()

                                        from src.utils.security import decrypt_data

                                        access_token = decrypt_data(
                                            account.encrypted_token
                                        )
                                        access_token_secret = decrypt_data(
                                            account.encrypted_token_secret
                                        )

                                        result = post_to_twitter(
                                            api_key,
                                            api_secret,
                                            access_token,
                                            access_token_secret,
                                            content,
                                            media_path,
                                        )

                                    elif platform == "facebook":
                                        from src.utils.security import decrypt_data

                                        access_token = decrypt_data(
                                            account.encrypted_token
                                        )

                                        result = post_to_facebook(
                                            access_token, content, media_path
                                        )

                                    elif platform == "linkedin":
                                        from src.utils.security import decrypt_data

                                        access_token = decrypt_data(
                                            account.encrypted_token
                                        )

                                        result = post_to_linkedin(
                                            access_token, content, media_path
                                        )

                                    else:
                                        result = {
                                            "success": False,
                                            "message": f"Posting to {platform} not implemented yet",
                                        }

                                    if result["success"]:
                                        st.success(
                                            f"Posted to {platform.capitalize()} successfully!"
                                        )
                                    else:
                                        st.error(
                                            f"Error posting to {platform.capitalize()}: {result['message']}"
                                        )

                                except Exception as e:
                                    st.error(
                                        f"Error posting to {platform.capitalize()}: {str(e)}"
                                    )

                    # Clean up temp file
                    if media_path and os.path.exists(media_path):
                        os.unlink(media_path)

    with post_tab2:
        # AI-generated post
        with st.form("ai_post_form"):
            # Post type
            post_type = st.selectbox(
                "Post Type",
                options=["portfolio_update", "market_update", "investment_tip"],
                format_func=lambda x: {
                    "portfolio_update": "Portfolio Update",
                    "market_update": "Market Update",
                    "investment_tip": "Investment Tip",
                }.get(x, x),
            )

            # Custom inputs based on post type
            if post_type == "portfolio_update":
                portfolio_value = st.number_input(
                    "Portfolio Value (USD)", value=10000.0, step=1000.0
                )
                daily_change = st.number_input("Daily Change (%)", value=1.5, step=0.1)
                top_assets = st.text_input(
                    "Top Performing Assets (comma-separated)", value="BTC, ETH, TSLA"
                )

                post_data = {
                    "portfolio_value": portfolio_value,
                    "daily_change": daily_change,
                    "top_assets": [a.strip() for a in top_assets.split(",")],
                }

            elif post_type == "market_update":
                # Market data
                market_data = {
                    "btc_price": st.number_input(
                        "Bitcoin Price (USD)", value=69420.0, step=100.0
                    ),
                    "eth_price": st.number_input(
                        "Ethereum Price (USD)", value=3500.0, step=10.0
                    ),
                    "market_trend": st.selectbox(
                        "Market Trend", options=["bullish", "bearish", "neutral"]
                    ),
                }
                sentiment = st.selectbox(
                    "Overall Sentiment", options=["positive", "negative", "neutral"]
                )

                post_data = {"market_data": market_data, "sentiment": sentiment}

            else:  # investment_tip
                tip_topic = st.selectbox(
                    "Tip Topic",
                    options=[
                        "diversification",
                        "dollar_cost_averaging",
                        "risk_management",
                        "emergency_fund",
                        "tax_efficiency",
                    ],
                )

                post_data = {"topic": tip_topic}

            # Generate button
            generate = st.form_submit_button("Generate Post")

            if generate:
                with st.spinner("Generating post with AI..."):
                    try:
                        # Generate post content
                        post_content = generate_social_post(post_type, post_data)

                        if post_content.startswith("Error:"):
                            st.error(post_content)
                        else:
                            st.success("Post generated!")

                            # Display the generated post
                            st.text_area(
                                "Generated Post", value=post_content, height=100
                            )

                            # Mock social preview
                            st.subheader("Social Preview")
                            with st.container(border=True):
                                st.markdown("**@YourHandle** · Just now")
                                st.markdown(post_content)

                            # Provide options to post
                            st.subheader("Post Options")

                            # Create columns for buttons
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("Post Now", key="post_ai_now"):
                                    st.info(
                                        "This would post the content immediately. In the full app, this would call the same posting function as the manual form."
                                    )

                            with col2:
                                if st.button("Schedule", key="schedule_ai"):
                                    st.info(
                                        "This would open a scheduling dialog. In the full app, this would call the same scheduling function as the manual form."
                                    )

                            with col3:
                                if st.button("Edit First", key="edit_ai"):
                                    # Set the content in the manual post form
                                    st.session_state[
                                        "manual_post_content"
                                    ] = post_content
                                    st.info(
                                        "In the full app, this would copy the content to the manual post form for editing."
                                    )

                    except Exception as e:
                        st.error(f"Error generating post: {str(e)}")


def show_scheduled_posts(user_id):
    """Show scheduled posts section."""
    st.subheader("Scheduled Posts")

    # Get scheduled posts
    result = get_scheduled_posts(user_id)

    if result["success"] and result["posts"]:
        for post in result["posts"]:
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    if post["platform"] == "twitter":
                        st.markdown("🐦 **Twitter**")
                    elif post["platform"] == "facebook":
                        st.markdown("📘 **Facebook**")
                    elif post["platform"] == "linkedin":
                        st.markdown("🔵 **LinkedIn**")
                    elif post["platform"] == "instagram":
                        st.markdown("📸 **Instagram**")

                    scheduled_time = datetime.fromisoformat(post["scheduled_time"])
                    st.caption(f"Scheduled for:")
                    st.caption(scheduled_time.strftime("%Y-%m-%d"))
                    st.caption(scheduled_time.strftime("%H:%M"))

                with col2:
                    st.markdown(
                        post["content"][:100]
                        + ("..." if len(post["content"]) > 100 else "")
                    )

                    if post["media_path"]:
                        st.caption("Includes media attachment")

                with col3:
                    if st.button("Cancel", key=f"cancel_{post['id']}"):
                        result = cancel_scheduled_post(post["id"])
                        if result["success"]:
                            st.success("Scheduled post cancelled")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Error cancelling post: {result['message']}")
    else:
        st.info("No scheduled posts found.")


def show_sentiment_analysis():
    """Show sentiment analysis section."""
    st.subheader("Social Media Sentiment Analysis")

    # Input for topic to analyze
    query = st.text_input(
        "Enter Topic to Analyze", placeholder="BTC, TSLA, #crypto, etc."
    )

    if st.button("Analyze Sentiment"):
        if not query:
            st.error("Please enter a topic to analyze")
            return

        with st.spinner("Analyzing social media sentiment..."):
            # In a real app, this would call Twitter API to get tweets
            # For demo, generate some mock data

            # Mock sentiment data
            sentiment_data = {
                "positive": 65,
                "neutral": 20,
                "negative": 15,
                "total_posts": 250,
                "trending_words": ["moon", "bullish", "invest", "future", "gains"],
                "notable_accounts": ["@elonmusk", "@CryptoAnalyst", "@TechInvestor"],
                "recent_posts": [
                    {
                        "text": f"Really bullish on {query} right now! The fundamentals are strong and the technical analysis looks promising. #investing",
                        "user": "CryptoAnalyst",
                        "likes": 432,
                        "sentiment": "positive",
                    },
                    {
                        "text": f"Just increased my position in {query}. The recent dip was a great buying opportunity.",
                        "user": "InvestorDaily",
                        "likes": 256,
                        "sentiment": "positive",
                    },
                    {
                        "text": f"Not sure about {query} at current prices. Might be overvalued considering market conditions.",
                        "user": "TechInvestor",
                        "likes": 187,
                        "sentiment": "neutral",
                    },
                    {
                        "text": f"Disappointed with the performance of {query} lately. Was expecting more after the recent announcements.",
                        "user": "StockTrader",
                        "likes": 92,
                        "sentiment": "negative",
                    },
                ],
            }

            # Display sentiment overview
            st.subheader(f"Sentiment Analysis for '{query}'")

            # Sentiment distribution chart
            fig = px.pie(
                values=[
                    sentiment_data["positive"],
                    sentiment_data["neutral"],
                    sentiment_data["negative"],
                ],
                names=["Positive", "Neutral", "Negative"],
                title="Sentiment Distribution",
                color_discrete_sequence=["#28a745", "#6c757d", "#dc3545"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

            # Key metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Posts Analyzed", sentiment_data["total_posts"])

            with col2:
                st.metric("Positive Sentiment", f"{sentiment_data['positive']}%")

            with col3:
                st.metric(
                    "Engagement Level",
                    "High" if sentiment_data["total_posts"] > 200 else "Medium",
                )

            # Trending words
            st.subheader("Trending Words")
            st.markdown(
                " • ".join(
                    [f"**#{word}**" for word in sentiment_data["trending_words"]]
                )
            )

            # Recent posts
            st.subheader("Sample Posts")
            for post in sentiment_data["recent_posts"]:
                with st.container(border=True):
                    st.markdown(f"**@{post['user']}**")
                    st.markdown(post["text"])

                    # Show sentiment indicator
                    if post["sentiment"] == "positive":
                        st.markdown("🟢 Positive • ❤️ " + str(post["likes"]))
                    elif post["sentiment"] == "neutral":
                        st.markdown("⚪ Neutral • ❤️ " + str(post["likes"]))
                    else:
                        st.markdown("🔴 Negative • ❤️ " + str(post["likes"]))

            # Summary
            st.subheader("AI Summary")
            st.markdown(
                f"""
                The overall sentiment for **{query}** is **mostly positive** (65%) with some neutral and negative opinions.
                Many users are expressing optimism about future growth potential. Common themes include:

                1. Strong fundamentals driving bullish sentiment
                2. Recent price movements being seen as a buying opportunity
                3. Some concerns about current valuation compared to the broader market

                Notable accounts like @CryptoAnalyst and @TechInvestor have been influential in the conversation.
                Based solely on social sentiment, there appears to be positive momentum building around {query}.
            """
            )


def show_social_media():
    """Display the Social Media page."""
    # Require login
    user = require_login()

    # Add page title
    st.title("Social Media Integration")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Connected Accounts", "Create Post", "Scheduled Posts", "Sentiment Analysis"]
    )

    with tab1:
        show_accounts_section(user["id"])

    with tab2:
        show_post_section(user["id"])

    with tab3:
        show_scheduled_posts(user["id"])

    with tab4:
        show_sentiment_analysis()
