import logging
from datetime import datetime
from typing import Dict, List, Union

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api.ai_analysis import analyze_with_best_model
from src.models.database import PortfolioOptimization, SessionLocal
from src.utils.auth import require_login
from src.utils.portfolio_optimizer import PortfolioOptimizer

# Make the lookback periods globally accessible
LOOKBACK_PERIODS = [
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "3y",
    "5y",
    "10y",
    "15y",
    "20y",
    "25y",
    "max",
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_portfolio_optimizer() -> None:
    """
    Display the Portfolio Optimization page with LLM-assisted analysis.
    """
    require_login()

    st.title("Portfolio Optimizer")
    st.write("Optimize your portfolio allocation with AI assistance.")

    # Create tabs
    tab1, tab2 = st.tabs(["📊 ETF Portfolio Optimization", "💬 Custom Analysis"])

    with tab1:
        show_etf_optimizer()

    with tab2:
        show_custom_analysis()


def show_etf_optimizer() -> None:
    """
    Display the ETF Portfolio Optimization interface
    """
    st.header("ETF Portfolio Optimization")
    st.write("Analyze optimal allocations for ETFs and leveraged ETFs")

    # User input for base ETF
    base_etf = st.text_input(
        "Base ETF Symbol (e.g., SPY, QQQ)",
        "SPY",
        help="Enter the ticker symbol for the base ETF. Supported ETFs include SPY (S&P 500), QQQ (Nasdaq), IWM (Russell 2000)",
    )

    # User input for leveraged ETFs
    st.write("Select leveraged ETF multipliers to include:")
    col1, col2, col3 = st.columns(3)

    with col1:
        include_3x = st.checkbox(
            "3x Leveraged (e.g., SPXL)",
            value=True,
            help="3x leveraged ETFs aim to deliver 3x the daily returns of the underlying index",
        )
        include_2x = st.checkbox(
            "2x Leveraged (e.g., SSO)",
            value=True,
            help="2x leveraged ETFs aim to deliver 2x the daily returns of the underlying index",
        )

    with col2:
        include_1x = st.checkbox(
            "1x (Base ETF)",
            value=True,
            help="The standard non-leveraged ETF that tracks the underlying index",
        )
        include_inverse = st.checkbox(
            "-1x Inverse (e.g., SH)",
            value=False,
            help="Inverse ETFs aim to deliver the opposite of the underlying index's daily returns",
        )

    with col3:
        include_inverse_2x = st.checkbox(
            "-2x Inverse (e.g., SDS)",
            value=False,
            help="2x inverse ETFs aim to deliver -2x the daily returns of the underlying index",
        )
        include_inverse_3x = st.checkbox(
            "-3x Inverse (e.g., SPXS)",
            value=False,
            help="3x inverse ETFs aim to deliver -3x the daily returns of the underlying index",
        )

    # Optimization parameters
    st.subheader("Optimization Parameters")
    lookback_period = st.select_slider(
        "Historical Lookback Period",
        options=LOOKBACK_PERIODS,
        value="1y",
        help="Longer periods provide more historical context but may have data availability issues. For best results with leveraged ETFs, use shorter periods (1y or less).",
    )

    optimization_method = st.radio(
        "Optimization Method",
        options=["Maximum Sharpe Ratio", "Minimum Volatility", "Maximum Return"],
        index=0,
        help="Maximum Sharpe Ratio balances risk and return, Minimum Volatility focuses on reducing risk, Maximum Return prioritizes historical returns",
    )

    # Add informational text
    st.markdown(
        """
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px">
    <small>This optimization analyzes historical ETF price data to determine optimal allocation weights based on your selected time period and optimization method.
    Results are for educational purposes and not financial advice.</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Add button to run analysis
    if st.button(
        "Generate Optimal Portfolio Allocation",
        help="Click to run the optimization with your selected parameters",
    ):
        with st.spinner("Analyzing optimal portfolio allocation..."):
            # Get the list of ETFs based on user selection
            etfs_to_include = get_etfs_list(
                base_etf,
                include_3x,
                include_2x,
                include_1x,
                include_inverse,
                include_inverse_2x,
                include_inverse_3x,
            )

            # Create a database session
            db = SessionLocal()

            try:
                # Run optimization with data caching - always prefer live data
                optimization_result = PortfolioOptimizer.optimize_portfolio(
                    base_etf=base_etf,
                    included_etfs=etfs_to_include,
                    lookback_period=lookback_period,
                    optimization_method=optimization_method,
                    use_cache=True,
                    prefer_live_data=False,  # Use cached data first to avoid API issues
                    save_result=True,
                    user_id=st.session_state.get("user", {}).get("id"),
                )

                # Check for errors
                if "error" in optimization_result:
                    error_msg = optimization_result["error"]
                    st.error(f"Error during optimization: {error_msg}")

                    # Provide helpful guidance based on the error
                    if (
                        "Could not retrieve data for any of the requested ETFs"
                        in error_msg
                    ):
                        symbols = (
                            error_msg.split("ETFs: ")[1]
                            if "ETFs: " in error_msg
                            else "the selected ETFs"
                        )
                        st.warning(
                            f"""
                        None of the selected ETFs could be retrieved: {symbols}

                        This might be due to:
                        1. Temporary API connection issues with Yahoo Finance, Alpha Vantage, and other providers
                        2. Rate limits on financial data APIs (especially if you've made many requests)
                        3. The symbols may not exist or may be delisted

                        Suggestions:
                        - Try again in a few minutes
                        - Try a different time period (shorter periods are more reliable)
                        - Verify the ETF symbols are correct and actively traded
                        - Select different ETFs to include in your portfolio
                        """
                        )
                    elif "Could not retrieve data" in error_msg:
                        symbol = (
                            error_msg.split("for ")[1].split(" from")[0]
                            if "for " in error_msg
                            else "ETF"
                        )
                        st.warning(
                            f"""
                        Data for {symbol} could not be retrieved despite trying multiple financial data sources. This might be due to:
                        1. Temporary API connection issues with Yahoo Finance, Alpha Vantage, and other providers
                        2. Rate limits on financial data APIs (especially if you've made many requests)
                        3. The symbol may not exist or may be delisted

                        Suggestions:
                        - Try again in a few minutes
                        - Try a different time period (shorter periods are more reliable)
                        - Check if the symbol is correct and actively traded
                        """
                        )
                    return

                # Prepare the prompt for LLM with the optimization results
                try:
                    prompt = create_optimization_prompt_with_results(
                        base_etf=base_etf,
                        etfs_to_include=etfs_to_include,
                        lookback_period=lookback_period,
                        optimization_method=optimization_method,
                        weights=optimization_result.get("weights", {}),
                        metrics=optimization_result.get("metrics", {}),
                    )
                except Exception as e:
                    logger.error(f"Error creating optimization prompt: {str(e)}")
                    # Fallback to basic prompt
                    prompt = f"""
                    Analyze a portfolio of {base_etf} and related ETFs optimized using {optimization_method}.

                    Due to data availability issues, only limited metrics could be calculated.
                    Discuss general principles of leveraged ETF portfolio optimization and potential risks.
                    """

                # Get LLM response
                try:
                    response = analyze_with_best_model(prompt, task_type="finance")
                    st.success("Successfully generated AI analysis")
                except Exception as ai_error:
                    st.error(f"Error generating AI analysis: {str(ai_error)}")
                    # Log details for debugging
                    logger.error(f"AI analysis error details: {str(ai_error)}")
                    response = "Error generating analysis. Please try again later."

                # Save the analysis text to the database
                if "id" in optimization_result:
                    # Update the saved optimization with the analysis text
                    optimization = (
                        db.query(PortfolioOptimization)
                        .filter(PortfolioOptimization.id == optimization_result["id"])
                        .first()
                    )
                    if optimization:
                        optimization.analysis_text = response
                        db.commit()

                # Display the analysis
                st.subheader("Portfolio Optimization Analysis")
                if isinstance(response, str):
                    st.write(response)
                else:
                    st.error("Failed to generate analysis. Please try again.")

                # Generate and display performance comparison chart
                with st.spinner("Generating performance comparison..."):
                    try:
                        # Add more explicit checks to avoid Series truth value errors
                        has_portfolio_perf = (
                            optimization_result is not None
                            and "portfolio_performance" in optimization_result
                        )
                        has_etfs_data = (
                            optimization_result is not None
                            and "etfs_data" in optimization_result
                        )
                        has_valid_etfs_data = False

                        if has_etfs_data and isinstance(
                            optimization_result["etfs_data"], dict
                        ):
                            # Check if etfs_data is not empty dict
                            has_valid_etfs_data = (
                                len(optimization_result["etfs_data"]) > 0
                            )

                        if has_portfolio_perf and has_etfs_data and has_valid_etfs_data:
                            try:
                                display_portfolio_results(
                                    optimization_result=optimization_result
                                )
                            except Exception as inner_e:
                                logger.error(
                                    f"Error in display_portfolio_results: {str(inner_e)}"
                                )
                                st.warning(
                                    "Could not display performance charts due to data format issues."
                                )
                                st.info(
                                    "The optimization was successful, but visualization failed. Try again with a different time period or ETFs."
                                )
                        else:
                            st.warning(
                                "Limited data available. Could not generate full performance comparison charts."
                            )
                            st.info(
                                "The analysis is based on the available data. For better results, try again later or select different ETFs."
                            )
                    except Exception as e:
                        logger.error(f"Error displaying results: {str(e)}")
                        st.warning("Could not display portfolio performance charts.")
                        st.info(
                            "You can still see the AI analysis above. Consider trying a different lookback period or ETF combination."
                        )

                # Create a disclaimer
                st.info(
                    "⚠️ Disclaimer: This analysis is for educational purposes only. "
                    "Past performance is not indicative of future results. "
                    "Always conduct your own research before making investment decisions."
                )

            except Exception as e:
                logger.error(f"Error in portfolio optimization: {str(e)}")
                st.error(f"Unable to complete portfolio optimization: {str(e)}")

            finally:
                # Always close the database session
                db.close()


def show_custom_analysis() -> None:
    """
    Display interface for custom portfolio analysis questions
    """
    st.header("Custom Portfolio Analysis")
    st.write("Ask specific questions about portfolio optimization and leveraged ETFs")

    # User input for custom question
    user_question = st.text_area(
        "Enter your question about portfolio optimization:",
        height=100,
        placeholder="Example: How would you optimize a portfolio of SPY, QQQ, and their leveraged versions to maximize the Sharpe ratio?",
        help="Ask questions about portfolio theory, ETF allocation strategies, or leveraged ETF combinations",
    )

    # Add example question section
    with st.expander("⭐ Example Questions", expanded=False):
        st.markdown(
            """
        - What is volatility decay and how does it affect leveraged ETFs?
        - How should I balance risk and reward when using leveraged ETFs?
        - What rebalancing frequency works best with leveraged ETF portfolios?
        - How do inverse ETFs perform in different market conditions?
        - What is the Kelly Criterion and how can I apply it to my portfolio?
        """
        )

    # Add button to submit question
    if (
        st.button(
            "Get AI Analysis", help="Click to generate analysis based on your question"
        )
        and user_question
    ):
        with st.spinner("Generating analysis..."):
            # Prepare prompt with the user's question
            prompt = f"""
            You are a financial advisor specializing in portfolio optimization and ETF analysis.

            Please answer the following question about portfolio optimization:

            {user_question}

            Provide a detailed analysis with reasoning and, if applicable, suggested allocation percentages.
            Include relevant financial considerations and risks, particularly when dealing with leveraged ETFs.
            """

            try:
                # Get LLM response
                response = analyze_with_best_model(prompt, task_type="finance")

                # Display the analysis
                st.subheader("Analysis")
                st.write(response)

                # Create a disclaimer
                st.info(
                    "⚠️ Disclaimer: This analysis is for educational purposes only. "
                    "Past performance is not indicative of future results. "
                    "Always conduct your own research before making investment decisions."
                )

            except Exception as e:
                logger.error(f"Error generating custom analysis: {str(e)}")
                st.error("Unable to generate analysis. Please try again later.")


def get_etfs_list(
    base_etf: str,
    include_3x: bool,
    include_2x: bool,
    include_1x: bool,
    include_inverse: bool,
    include_inverse_2x: bool,
    include_inverse_3x: bool,
) -> List[str]:
    """
    Generate a list of ETF symbols based on user selection

    Args:
        base_etf: The base ETF symbol (e.g., SPY)
        include_*: Boolean flags for which leveraged versions to include

    Returns:
        List of ETF symbols
    """
    # Determine which ETFs to include
    etfs_to_include = []

    # Map common ETFs and their leveraged versions
    etf_mapping = {
        "SPY": {
            "3x": "SPXL",
            "2x": "SSO",
            "1x": "SPY",
            "-1x": "SH",
            "-2x": "SDS",
            "-3x": "SPXS",
        },
        "QQQ": {
            "3x": "TQQQ",
            "2x": "QLD",
            "1x": "QQQ",
            "-1x": "PSQ",
            "-2x": "QID",
            "-3x": "SQQQ",
        },
        "IWM": {
            "3x": "TNA",
            "2x": "UWM",
            "1x": "IWM",
            "-1x": "RWM",
            "-2x": "TWM",
            "-3x": "TZA",
        },
        # Add more mappings as needed
    }

    # Default mapping if the base ETF isn't in our dictionary
    default_mapping = {
        "3x": f"{base_etf} 3x",
        "2x": f"{base_etf} 2x",
        "1x": base_etf,
        "-1x": f"{base_etf} -1x",
        "-2x": f"{base_etf} -2x",
        "-3x": f"{base_etf} -3x",
    }

    # Get the appropriate mapping
    mapping = etf_mapping.get(base_etf.upper(), default_mapping)

    # Add ETFs based on user selection, check for valid symbols
    if include_3x and mapping["3x"] and mapping["3x"] != f"{base_etf} 3x":
        etfs_to_include.append(mapping["3x"])
    if include_2x and mapping["2x"] and mapping["2x"] != f"{base_etf} 2x":
        etfs_to_include.append(mapping["2x"])
    if include_1x:
        etfs_to_include.append(mapping["1x"])
    if include_inverse and mapping["-1x"] and mapping["-1x"] != f"{base_etf} -1x":
        etfs_to_include.append(mapping["-1x"])
    if include_inverse_2x and mapping["-2x"] and mapping["-2x"] != f"{base_etf} -2x":
        etfs_to_include.append(mapping["-2x"])
    if include_inverse_3x and mapping["-3x"] and mapping["-3x"] != f"{base_etf} -3x":
        etfs_to_include.append(mapping["-3x"])

    return etfs_to_include


def create_optimization_prompt_with_results(
    base_etf: str,
    etfs_to_include: List[str],
    lookback_period: str,
    optimization_method: str,
    weights: Dict[str, float],
    metrics: Dict[str, Dict[str, Union[float, str]]],
) -> str:
    """
    Create a prompt for the LLM to analyze portfolio allocation with optimization results

    Args:
        base_etf: The base ETF symbol (e.g., SPY)
        etfs_to_include: List of ETF symbols included in the optimization
        lookback_period: Historical data period considered
        optimization_method: Method used for optimization
        weights: Dictionary of optimized weights
        metrics: Dictionary of performance metrics

    Returns:
        Formatted prompt for LLM analysis
    """
    # Format ETFs for the prompt
    etfs_formatted = ", ".join(etfs_to_include)

    # Format the weights for the prompt
    weights_formatted = "\n".join(
        [f"- {symbol}: {weights[symbol]*100:.2f}%" for symbol in weights]
    )

    # Format portfolio metrics
    portfolio_metrics = metrics.get("Optimal Portfolio", {})

    # Create the prompt with actual optimization results
    prompt = f"""
    As a financial portfolio expert, analyze the following optimized portfolio allocation for {base_etf} and its leveraged variants: {etfs_formatted}.

    OPTIMIZATION PARAMETERS:
    - Historical lookback period: {lookback_period}
    - Optimization method: {optimization_method}

    OPTIMIZED PORTFOLIO WEIGHTS:
    {weights_formatted}

    PORTFOLIO PERFORMANCE METRICS:
    - Total Return: {portfolio_metrics.get('total_return_fmt', 'N/A')}
    - Annualized Return: {portfolio_metrics.get('annualized_return_fmt', 'N/A')}
    - Volatility: {portfolio_metrics.get('volatility_fmt', 'N/A')}
    - Maximum Drawdown: {portfolio_metrics.get('max_drawdown_fmt', 'N/A')}
    - Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio_fmt', 'N/A')}

    Provide a comprehensive analysis addressing:
    1. How the optimized weights align with the chosen optimization method ({optimization_method})
    2. Explain why these specific weights were chosen and how they balance risk and return
    3. Discuss the mathematical and financial principles behind this optimization approach
    4. What are the key risks of this allocation, particularly with leveraged ETFs?
    5. How investors should implement this strategy, including rebalancing frequency and monitoring

    Consider volatility decay, correlation effects, and market regime changes in your analysis.
    Provide examples of how this portfolio might perform in different market conditions.
    """

    return prompt


def create_optimization_prompt(
    base_etf: str,
    include_3x: bool,
    include_2x: bool,
    include_1x: bool,
    include_inverse: bool,
    include_inverse_2x: bool,
    include_inverse_3x: bool,
    lookback_period: str,
    optimization_method: str,
) -> str:
    """
    Create a prompt for the LLM to analyze optimal portfolio allocation

    Args:
        base_etf: The base ETF symbol (e.g., SPY)
        include_*: Boolean flags for which leveraged versions to include
        lookback_period: Historical data period to consider
        optimization_method: Method for optimization

    Returns:
        Formatted prompt for LLM analysis
    """
    # Get ETFs to include
    etfs_to_include = get_etfs_list(
        base_etf,
        include_3x,
        include_2x,
        include_1x,
        include_inverse,
        include_inverse_2x,
        include_inverse_3x,
    )

    # Format ETFs for the prompt
    etfs_formatted = ", ".join(etfs_to_include)

    # Create the prompt
    prompt = f"""
    As a financial portfolio expert, analyze the optimal allocation strategy for a portfolio composed of {base_etf} and its leveraged variants: {etfs_formatted}.

    Consider the following:
    - Historical lookback period: {lookback_period}
    - Optimization method: {optimization_method}

    Specific questions to address:
    1. What would be the optimal weight vector for these ETFs to maximize future expected return?
    2. How would you use backtesting to determine these optimal weights?
    3. Explain the mathematical and financial principles behind this optimization.
    4. What are the risks of this approach, particularly with leveraged ETFs?
    5. Provide a suggested percentage allocation across the selected ETFs.

    Consider volatility decay, rebalancing frequency, and correlation effects in your analysis.
    Provide examples of how this portfolio might perform in different market conditions.
    """

    return prompt


def convert_period_to_days(period: str) -> int:
    """Convert period string to days for data fetching"""
    if period.endswith("y"):
        return int(period[:-1]) * 365
    elif period.endswith("mo"):
        return int(period[:-2]) * 30
    elif period.endswith("d"):
        return int(period[:-1])
    else:
        # Default to 1 year if unknown format
        return 365


# This function is replaced by PortfolioOptimizer.get_etf_price_history


# This function is replaced by PortfolioOptimizer.calculate_optimal_weights


# This function is replaced by PortfolioOptimizer.calculate_portfolio_performance


def display_portfolio_results(optimization_result: Dict) -> None:
    """
    Display performance charts and metrics from optimization results

    Args:
        optimization_result: Dictionary containing optimization results
    """
    # Extract data from optimization result
    etfs_data = optimization_result.get("etfs_data", {})
    weights = optimization_result.get("weights", {})
    metrics = optimization_result.get("metrics", {})
    weights_display = optimization_result.get("weights_display", {})
    portfolio_performance = optimization_result.get("portfolio_performance", None)
    lookback_period = optimization_result.get("lookback_period", "1y")

    # Add more detailed validations
    if not isinstance(etfs_data, dict) or len(etfs_data) == 0:
        st.warning("No ETF data available for visualization.")
        return

    if not isinstance(weights, dict) or len(weights) == 0:
        st.warning("No weight data available for visualization.")
        return

    if portfolio_performance is None:
        st.warning("No portfolio performance data available for visualization.")
        # We can continue with just ETF data

    st.subheader("Performance Comparison")

    # Initialize session state for chart lookback if not present
    if "chart_lookback" not in st.session_state:
        st.session_state.chart_lookback = lookback_period

    # Allow user to select a different lookback period for visualization without page rerun
    new_lookback = st.select_slider(
        "Chart Lookback Period",
        options=LOOKBACK_PERIODS,
        value=st.session_state.chart_lookback,
        key="chart_lookback_selector",
        on_change=lambda: None,  # Prevent automatic rerun when slider changes
    )

    # Update session state with the selected value
    st.session_state.chart_lookback = new_lookback

    if new_lookback != lookback_period:
        st.info(
            f"Viewing data with {new_lookback} lookback. To recalculate optimization with this period, please return to the main form and adjust the setting there."
        )

    # Display the weights in an expander
    with st.expander("Optimal Portfolio Weights"):
        weights_df = pd.DataFrame(
            {
                "ETF": weights.keys(),
                "Weight": [f"{v*100:.1f}%" for v in weights.values()],
            }
        )
        st.dataframe(weights_df, hide_index=True)

    # If the user selected a different lookback period, try to fetch that data
    if new_lookback != lookback_period:
        try:
            # Create database session
            db = SessionLocal()

            # Fetch historical data for each ETF with the new lookback period
            new_etfs_data = {}
            for symbol in etfs_data.keys():
                try:
                    # Get data with preference for cached data
                    price_series = PortfolioOptimizer.get_etf_price_history(
                        db=db,
                        symbol=symbol,
                        lookback_period=new_lookback,
                        use_cache=True,
                        prefer_live_data=False,  # Use cached data first
                    )
                    new_etfs_data[symbol] = price_series
                except Exception as e:
                    st.warning(
                        f"Could not load {new_lookback} data for {symbol}: {str(e)}"
                    )

            # If we got data, use it
            if new_etfs_data:
                etfs_data = new_etfs_data

                # Calculate portfolio performance with the new data
                try:
                    new_portfolio_performance = (
                        PortfolioOptimizer.calculate_portfolio_performance(
                            etfs_data=new_etfs_data, weights=weights
                        )
                    )
                    portfolio_performance = new_portfolio_performance
                except Exception as e:
                    logger.error(f"Error calculating portfolio performance: {str(e)}")
                    st.warning(
                        "Using sample performance data. Try different ETFs or time periods."
                    )

                    # Create sample portfolio performance for display purposes
                    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
                    base_value = 100.0

                    # Generate slightly upward trending portfolio values
                    values = []
                    for i in range(len(dates)):
                        day_change = 0.001 * (i % 5) + 0.0005  # Small upward trend
                        value = base_value * (1 + day_change * i)
                        values.append(value)

                    portfolio_performance = pd.Series(values, index=dates)
            else:
                st.warning(
                    f"Could not load {new_lookback} data for any ETFs. Using original data."
                )
                portfolio_performance = optimization_result.get("portfolio_performance")

            # Close the database session
            db.close()
        except Exception as e:
            st.warning(
                f"Error loading {new_lookback} data: {str(e)}. Using original data."
            )
            portfolio_performance = optimization_result.get("portfolio_performance")
    else:
        # Use the portfolio performance directly from the optimization result
        portfolio_performance = optimization_result.get("portfolio_performance")

    # Create normalized performance data for plotting with additional validation
    normalized_data = {}
    valid_data_found = False

    # Log the etfs_data structure to help with debugging
    logger.info(
        f"ETFs data structure: {type(etfs_data)}, keys: {list(etfs_data.keys())}"
    )

    for symbol, series in etfs_data.items():
        # Skip if series is empty or None
        if series is None or len(series) < 2:  # Need at least 2 points for a chart
            logger.warning(
                f"Skipping {symbol} - insufficient data points: {len(series) if series is not None else 'None'}"
            )
            continue

        # Use .values to avoid ambiguity in boolean context
        try:
            # Log the series information
            logger.info(
                f"Processing {symbol} series: type={type(series)}, index_type={type(series.index)}, length={len(series)}"
            )

            # Check for valid values
            if series.isna().all():
                logger.warning(f"Skipping {symbol} - all values are NaN")
                continue

            # Handle both DataFrame and Series cases
            # Force conversion to Series
            if isinstance(series, pd.DataFrame):
                # Try to get Close column first
                if "Close" in series.columns:
                    # Create a new Series with the same index but Close values
                    series_values = pd.Series(
                        series["Close"].values, index=series.index
                    )
                else:
                    # Use first column values with the original index
                    series_values = pd.Series(
                        series.iloc[:, 0].values, index=series.index
                    )
            else:
                # Already a Series
                series_values = series.copy()

            # Get first value safely
            if not series_values.empty:
                # Use .iloc[0].item() to avoid FutureWarning from float(series.iloc[0])
                try:
                    first_value = (
                        series_values.dropna().iloc[0].item()
                        if not series_values.dropna().empty
                        else 0
                    )
                except:
                    # Fallback if .item() doesn't work
                    first_value = (
                        float(series_values.dropna().iloc[0])
                        if not series_values.dropna().empty
                        else 0
                    )

                if first_value > 0:
                    # Ensure the series has properly formatted datetime index
                    if not isinstance(series.index, pd.DatetimeIndex):
                        try:
                            # Log original index for debugging
                            logger.info(f"{symbol} index sample: {series.index[:5]}")

                            # Convert index to datetime if it's not already
                            datetime_index = pd.to_datetime(
                                series.index, errors="coerce"
                            )

                            # Skip if all dates are in 1970 (epoch/parsing error)
                            if (
                                not datetime_index.empty
                                and datetime_index.min().year <= 1970
                            ):
                                logger.warning(
                                    f"Skipping {symbol} - invalid dates detected (epoch)"
                                )
                                continue

                            # Create new series with datetime index
                            series_values = pd.Series(
                                series_values.values, index=datetime_index
                            )
                            logger.info(
                                f"{symbol} converted to DatetimeIndex: {series_values.index[:5]}"
                            )
                        except Exception as date_error:
                            logger.warning(
                                f"Error converting {symbol} dates: {str(date_error)}"
                            )
                            continue  # Skip series with date conversion errors
                    else:
                        # Make sure series_values has the same index as the original series
                        series_values.index = series.index

                    # Normalize the series
                    normalized_series = series_values / first_value

                    # Check for valid values after normalization
                    if normalized_series.isna().all():
                        logger.warning(
                            f"Skipping {symbol} - normalization resulted in all NaN values"
                        )
                        continue

                    normalized_data[symbol] = normalized_series
                    valid_data_found = True
                else:
                    logger.warning(
                        f"Skipping {symbol} - first value is zero or negative: {first_value}"
                    )
            else:
                logger.warning(f"Skipping {symbol} - empty series")
        except Exception as e:
            logger.warning(f"Error normalizing {symbol} data: {str(e)}")

    # Add the portfolio performance to the data if available
    if portfolio_performance is not None and len(portfolio_performance) > 0:
        try:
            # Log the portfolio performance information
            logger.info(
                f"Portfolio performance: type={type(portfolio_performance)}, index_type={type(portfolio_performance.index)}, length={len(portfolio_performance)}"
            )

            # Check for valid values
            if portfolio_performance.isna().all():
                logger.warning(f"Skipping portfolio performance - all values are NaN")
            else:
                # Get first value with handling for NaN values
                first_value = float(portfolio_performance.dropna().iloc[0])

                if first_value > 0:
                    # Ensure the series has properly formatted datetime index
                    if not isinstance(portfolio_performance.index, pd.DatetimeIndex):
                        try:
                            # Log original index for debugging
                            logger.info("Analyzing portfolio index")

                            # Convert index to datetime if it's not already
                            datetime_index = pd.to_datetime(portfolio_performance.index)

                            # Skip if all dates are in 1970 (epoch/parsing error)
                            if datetime_index.min().year <= 1970:
                                logger.warning(
                                    f"Skipping portfolio performance - invalid dates detected (epoch)"
                                )
                            else:
                                # Create new series with datetime index
                                portfolio_performance = pd.Series(
                                    portfolio_performance.values, index=datetime_index
                                )
                                logger.info("Portfolio converted to DatetimeIndex")
                        except Exception as date_error:
                            logger.warning(
                                f"Error converting portfolio performance dates: {str(date_error)}"
                            )

                    # Add normalized portfolio performance if we have valid date index
                    if (
                        isinstance(portfolio_performance.index, pd.DatetimeIndex)
                        and portfolio_performance.index.min().year > 1970
                    ):
                        normalized_data["Optimal Portfolio"] = (
                            portfolio_performance / first_value
                        )
                        valid_data_found = True
        except Exception as e:
            logger.warning(f"Error normalizing portfolio performance: {str(e)}")

    if not valid_data_found:
        st.warning(
            "No valid data found for the performance comparison chart. Try a different lookback period or ETF selection."
        )

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(normalized_data)

    # Initialize chart tab selection in session state
    if "chart_tab_index" not in st.session_state:
        st.session_state.chart_tab_index = 0

    # Create tabs for different visualization options
    chart_tab_names = [
        "📈 Interactive Chart",
        "📊 Performance Metrics",
        "📋 Comparison Table",
    ]
    chart_tabs = st.tabs(chart_tab_names)

    with chart_tabs[0]:
        # Create a toggle for showing/hiding individual ETFs
        st.write("Toggle ETFs to display:")
        cols = st.columns(3)

        # Initialize etf visibility in session state if not present
        if "etf_visibility" not in st.session_state:
            st.session_state.etf_visibility = {}

        show_etfs = {}

        # Get ETF list from the normalized data keys
        available_etfs = [
            etf for etf in normalized_data.keys() if etf != "Optimal Portfolio"
        ]

        # Initialize visibility for new ETFs
        for etf in available_etfs:
            if etf not in st.session_state.etf_visibility:
                st.session_state.etf_visibility[etf] = True

        # Create a checkbox for each ETF
        for i, etf in enumerate(available_etfs):
            col_idx = i % 3
            with cols[col_idx]:
                # Create checkbox and read value from session state
                show_etfs[etf] = st.checkbox(
                    etf,
                    value=st.session_state.etf_visibility[etf],
                    key=f"etf_toggle_{etf}",
                )

                # Update the visibility state based on the checkbox value
                st.session_state.etf_visibility[etf] = show_etfs[etf]

        # Create the interactive chart
        fig = go.Figure()

        # Completely rebuild the plot data to ensure consistent datetime format
        try:
            # Create a fresh DataFrame to hold aligned time series
            aligned_data = {}
            common_index = None

            # Log data structure info for debugging
            logger.info(
                f"Plot data structure: {type(plot_data)}, columns: {plot_data.columns}"
            )

            # First, make sure all individual series have proper datetime indices
            for symbol, series in normalized_data.items():
                # Skip empty series
                if len(series) == 0:
                    continue

                try:
                    # Get the original index
                    original_index = series.index

                    # Convert to datetime if not already
                    if not isinstance(original_index, pd.DatetimeIndex):
                        # Try to parse the dates properly
                        datetime_index = pd.to_datetime(original_index)
                        # Create a new series with the datetime index
                        corrected_series = pd.Series(
                            series.values, index=datetime_index
                        )
                    else:
                        corrected_series = series.copy()

                    # Skip series with invalid dates (1970-01-01 is often a sign of parsing failure)
                    if corrected_series.index.min().year <= 1970:
                        logger.warning(
                            f"Skipping {symbol} - detected invalid dates (epoch)"
                        )
                        continue

                    # Store the properly dated series
                    aligned_data[symbol] = corrected_series

                    # Track the index for later alignment
                    if common_index is None:
                        common_index = corrected_series.index
                except Exception as e:
                    logger.warning(f"Error processing series for {symbol}: {str(e)}")

            # If we have valid series with datetime indices
            if aligned_data and common_index is not None:
                # Create a new DataFrame with all properly formatted series
                plot_data = pd.DataFrame(aligned_data)

                # Check for valid data again
                if plot_data.empty or len(plot_data.columns) == 0:
                    st.warning(
                        "No valid data series found for plotting after date processing."
                    )
                elif plot_data.index.min().year <= 1970:
                    st.warning(
                        "Invalid date ranges detected. Performance chart may not display correctly."
                    )
                    # Log the actual dates for debugging
                    logger.warning(
                        f"Date range: {plot_data.index.min()} to {plot_data.index.max()}"
                    )
            else:
                st.warning("Could not properly format any data series for plotting.")
        except Exception as e:
            logger.error(f"Error rebuilding plot data: {str(e)}")
            st.warning(
                "Error preparing chart data. Performance comparison may not display correctly."
            )

        # Add traces for each ETF if selected
        for etf, show in show_etfs.items():
            if show and etf in plot_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index, y=plot_data[etf], mode="lines", name=etf
                    )
                )

        # Always add the optimal portfolio if it exists in the data
        if "Optimal Portfolio" in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data["Optimal Portfolio"],
                    mode="lines",
                    name="Optimal Portfolio",
                    line=dict(width=3, dash="solid"),
                )
            )

        # Check if we have any traces in the figure
        if len(fig.data) > 0:
            # Update layout with proper date formatting
            fig.update_layout(
                title=f"Performance Comparison ({new_lookback} lookback)",
                xaxis_title="Date",
                yaxis_title="Normalized Return (Starting at 1.0)",
                legend_title="ETFs",
                hovermode="x unified",
                xaxis=dict(
                    type="date",
                    tickformat="%Y-%m",  # Format to show year-month
                    dtick="M3",  # Tick every 3 months
                    showgrid=True,
                    tickangle=-45,
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    showgrid=True,
                    zeroline=True,
                ),
            )
        else:
            # No valid traces, create a message in the chart area
            fig.add_annotation(
                text="No valid data available for performance comparison",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(
                title=f"Performance Comparison ({new_lookback} lookback)",
                xaxis_title="Date",
                yaxis_title="Normalized Return (Starting at 1.0)",
            )

        st.plotly_chart(fig, use_container_width=True)

    with chart_tabs[1]:
        # Calculate performance metrics
        metrics = {}

        for symbol, series in normalized_data.items():
            try:
                # Calculate returns (with error handling)
                if len(series) > 1:
                    returns = series.pct_change().dropna()

                    # Calculate metrics with safety checks
                    if len(series) > 0:
                        first_value = float(series.iloc[0])
                        last_value = float(series.iloc[-1])
                        if first_value > 0:
                            total_return = ((last_value / first_value) - 1) * 100
                        else:
                            total_return = 0
                    else:
                        total_return = 0

                    # Avoid division by zero or log of negative/zero values
                    if total_return <= -100:
                        annualized_return = -1.0  # -100% annual return
                    else:
                        try:
                            # Try to get days difference from datetime index
                            if hasattr(series.index[-1], "days") and hasattr(
                                series.index[0], "days"
                            ):
                                days = (series.index[-1] - series.index[0]).days
                            else:
                                # Handle case where index is string dates or integers
                                try:
                                    # Convert string dates to datetime if needed
                                    start_date = pd.to_datetime(series.index[0])
                                    end_date = pd.to_datetime(series.index[-1])
                                    days = (end_date - start_date).days
                                except Exception:
                                    # If conversion fails, estimate based on length of series
                                    # Assume daily data
                                    days = len(series)

                            if days > 0:
                                annualized_return = (
                                    (1 + total_return / 100) ** (365 / days)
                                ) - 1
                            else:
                                annualized_return = 0
                        except Exception as e:
                            logger.warning(
                                f"Error calculating annualized return: {str(e)}"
                            )
                            # Fallback - use simple estimate
                            annualized_return = total_return / 100

                    # Calculate volatility with safety checks
                    if len(returns) > 0:
                        volatility = returns.std() * (252**0.5) * 100  # Annualized
                    else:
                        volatility = 0

                    # Calculate max drawdown with safety checks
                    if len(returns) > 0:
                        cumulative = (1 + returns).cumprod()
                        if len(cumulative) > 0:
                            cummax = cumulative.cummax()
                            try:
                                # Check if all values in cummax are greater than 0
                                if cummax.gt(
                                    0
                                ).all():  # Explicit comparison instead of cummax > 0
                                    drawdowns = cumulative / cummax - 1
                                    max_dd = drawdowns.min() * 100
                                else:
                                    max_dd = 0
                            except Exception as e:
                                logger.warning(
                                    f"Error calculating max drawdown: {str(e)}"
                                )
                                max_dd = 0
                        else:
                            max_dd = 0
                    else:
                        max_dd = 0

                    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
                    risk_free = 0.02
                    if volatility > 0:
                        sharpe = (annualized_return - risk_free) / (volatility / 100)
                    else:
                        sharpe = 0

                    # Format metrics for display
                    metrics[symbol] = {
                        "Total Return (%)": f"{total_return:.2f}%",
                        "Annualized Return (%)": f"{annualized_return*100:.2f}%",
                        "Volatility (%)": f"{volatility:.2f}%",
                        "Max Drawdown (%)": f"{max_dd:.2f}%",
                        "Sharpe Ratio": f"{sharpe:.2f}",
                    }
                else:
                    # Not enough data points
                    metrics[symbol] = {
                        "Total Return (%)": "N/A",
                        "Annualized Return (%)": "N/A",
                        "Volatility (%)": "N/A",
                        "Max Drawdown (%)": "N/A",
                        "Sharpe Ratio": "N/A",
                    }
            except Exception as e:
                logger.warning(f"Error calculating metrics for {symbol}: {str(e)}")
                metrics[symbol] = {
                    "Total Return (%)": "Error",
                    "Annualized Return (%)": "Error",
                    "Volatility (%)": "Error",
                    "Max Drawdown (%)": "Error",
                    "Sharpe Ratio": "Error",
                }

        # Create and display the metrics table
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df)

        # Add a description of the metrics
        with st.expander("Metrics Explanation"):
            st.markdown(
                """
            - **Total Return**: The percentage change from the beginning to the end of the period
            - **Annualized Return**: The return expressed as an annual percentage
            - **Volatility**: The standard deviation of returns, annualized
            - **Max Drawdown**: The maximum percentage decline from a peak
            - **Sharpe Ratio**: The risk-adjusted return (assumes 2% risk-free rate)
            """
            )

    with chart_tabs[2]:
        # Create a comparison table with key data points
        comparison_data = {}

        for symbol, series in normalized_data.items():
            try:
                if len(series) > 0:
                    # Calculate start/end values and return with safety checks
                    try:
                        # Use .item() to avoid FutureWarning
                        start_value = series.iloc[0].item()
                        end_value = series.iloc[-1].item()
                    except Exception:
                        # Fallback method
                        start_value = float(series.iloc[0])
                        end_value = float(series.iloc[-1])

                    if start_value > 0:  # Avoid division by zero
                        total_return = ((end_value / start_value) - 1) * 100
                    else:
                        total_return = 0

                    # Calculate Sharpe ratio for this symbol from the metrics dictionary
                    sharpe_ratio = "N/A"
                    if symbol in metrics and "Sharpe Ratio" in metrics[symbol]:
                        sharpe_ratio = metrics[symbol]["Sharpe Ratio"]

                    # Add to comparison data
                    comparison_data[symbol] = {
                        "Starting Value": f"{start_value:.2f}",
                        "Ending Value": f"{end_value:.2f}",
                        "Total Return": f"{total_return:.2f}%",
                        "Sharpe Ratio": sharpe_ratio,
                        "Weight in Portfolio": weights_display.get(symbol, "N/A"),
                    }
                else:
                    # Handle empty series
                    comparison_data[symbol] = {
                        "Starting Value": "N/A",
                        "Ending Value": "N/A",
                        "Total Return": "N/A",
                        "Sharpe Ratio": "N/A",
                        "Weight in Portfolio": weights_display.get(symbol, "N/A"),
                    }
            except Exception as e:
                logger.warning(
                    f"Error calculating comparison data for {symbol}: {str(e)}"
                )
                comparison_data[symbol] = {
                    "Starting Value": "Error",
                    "Ending Value": "Error",
                    "Total Return": "Error",
                    "Sharpe Ratio": "Error",
                    "Weight in Portfolio": weights_display.get(symbol, "N/A"),
                }

        # Create and display the comparison table
        comparison_df = pd.DataFrame(comparison_data).T
        st.dataframe(comparison_df)
