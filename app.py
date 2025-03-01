import streamlit as st
from dotenv import load_dotenv
from src.components.navigation import sidebar_navigation
from src.utils.auth import handle_oauth_callback

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Wallet & Trend Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Handle OAuth callback if present
    handle_oauth_callback()

    # Display app title
    st.title("Crypto Wallet & Trend Analysis")

    # Get current page from navigation sidebar
    page = sidebar_navigation()

    # Import and display the selected page
    if page == "Dashboard":
        from src.pages.dashboard import show_dashboard

        show_dashboard()
    elif page == "Wallets":
        from src.pages.wallets import show_wallets

        show_wallets()
    elif page == "Trade/Invest":
        from src.pages.trade_invest import show_trade_invest

        show_trade_invest()
    elif page == "AI Insights":
        from src.pages.ai_insights import show_ai_insights

        show_ai_insights()
    elif page == "Portfolio Optimizer":
        from src.pages.portfolio_optimizer import show_portfolio_optimizer

        show_portfolio_optimizer()
    elif page == "Social Media":
        from src.pages.social_media import show_social_media

        show_social_media()
    elif page == "Settings":
        from src.pages.settings import show_settings

        show_settings()


if __name__ == "__main__":
    main()
