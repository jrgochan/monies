import streamlit as st


def sidebar_navigation():
    """
    Creates the sidebar navigation and returns the selected page.
    """
    # Display user info if logged in
    if "user" in st.session_state:
        st.sidebar.write(f"👋 Hello, {st.session_state.user['username']}")

    # Navigation options
    st.sidebar.title("Navigation")

    # Define pages with icons
    pages = {
        "Dashboard": "📊",
        "Wallets": "👛",
        "Trade/Invest": "💸",
        "AI Insights": "🧠",
        "Social Media": "📱",
        "Settings": "⚙️",
    }

    # Simple default navigation - safest option
    page = st.sidebar.radio(
        "Go to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}",
    )

    # Add logout button if user is logged in
    if "user" in st.session_state:
        if st.sidebar.button("Logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Add login/register links if no user is logged in
    else:
        st.sidebar.info("Please login to access all features")

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Crypto Wallet & Trend Analysis")

    return page
