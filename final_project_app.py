import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    joint = pd.read_csv("joint_metrics.csv", parse_dates=['MONTH_YEAR'])
    cbsa = pd.read_csv("cbsa_metrics.csv", parse_dates=['MONTH_YEAR'])
    # Filter to only include rows where MONTH_YEAR < 2025-01
    joint = joint[joint['MONTH_YEAR'] < pd.Timestamp('2025-01-01')]
    cbsa = cbsa[cbsa['MONTH_YEAR'] < pd.Timestamp('2025-01-01')]
    return joint, cbsa


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100


joint_df, cbsa_df = load_data()

st.title("Real Estate Market Time Series Analysis")
st.markdown("---")

# Project Documentation
with st.expander("Project Documentation", expanded=False):
    st.markdown("""
    ## Project Description: Real Estate Market Time Series Analysis

    This application performs comprehensive time series analysis on real estate market data across different Core Based Statistical Areas (CBSAs). The analysis includes:

    ### Key Features:
    - **Geographic Analysis**: Data from different CBSAs (metropolitan areas)
    - **Property Classification**: Option to analyze by property class (residential, commercial, etc.)
    - **Temporal Filtering**: Focus on specific time periods
    - **Multivariate Analysis**: VAR modeling for forecasting multiple metrics simultaneously

    ### Metrics Analyzed:
    - **PCT_LISTINGS_CLOSED**: Percentage of listings that closed successfully
    - **AVG_DAYS_ON_MARKET**: Average time properties spend on the market
    - **MEDIAN_CLOSE_PRICE**: Median closing price of properties
    - **AVG_CLOSE_TO_LIST_RATIO**: Ratio of closing price to listing price
    - **PCT_CHANGE_IN_LISTINGS**: Month-over-month change in listing volume

    ### Analysis Methods:
    1. **Descriptive Statistics**: Summary measures and trends
    2. **Stationarity Testing**: ADF and KPSS tests to determine if differencing is needed
    3. **ACF/PACF Analysis**: Autocorrelation and partial autocorrelation for ARIMA identification
    4. **Differencing**: First and seasonal differencing to achieve stationarity
    5. **VAR Modeling**: Vector Autoregression for multivariate forecasting
    6. **Forecast Evaluation**: RMSE and residual analysis

    This lab demonstrates practical application of time series econometrics in real estate market analysis.
    """)

# Step 1: Data Selection
st.header("Step 1: Data Selection")

col1, col2 = st.columns(2)
with col1:
    include_property_class = st.checkbox("Include Property Class", value=False)
with col2:
    # Determine dataset
    active_df = joint_df if include_property_class else cbsa_df
    
    # CBSA dropdown
    cbsa_options = active_df['CBSA_TITLE'].unique()
    selected_cbsa = st.selectbox("Select CBSA", sorted(cbsa_options))

# Property class selection (if applicable)
property_class_descriptions = {
    '0': 'Homes with more bathrooms (> 2.5), younger age (≤ 37.5 years), and smaller size (log sq.ft ≤ 8.25).',
    '1': 'Homes with more than 3.5 bedrooms, smaller size (≤ 3159.5 sq.ft), and newer age (≤ 39.5 years).',
    '3': 'Homes with more than 3.5 bedrooms, smaller size (≤ 3159.5 sq.ft), and older age (> 39.5 years), or homes with fewer bathrooms, older age (between 37.5 and 74.5 years), and more than 3.5 bedrooms.',
    '4': 'Homes with fewer bathrooms (≤ 2.5) and very old age (> 74.5 years).',
    '5': 'Homes with more bathrooms (> 2.5), younger age (≤ 37.5 years), and medium size (log sq.ft between 8.25 and 10.31).',
    '6': 'Homes with more bathrooms (> 2.5), younger age (≤ 37.5 years), and very large size (log sq.ft > 10.31).',
    '7': 'Homes with more than 3.5 bedrooms and larger size (> 3159.5 sq.ft).',
    '8': 'Homes with 3.5 bedrooms or fewer.',
    '9': 'Homes with fewer bathrooms (≤ 2.5), older age (between 37.5 and 74.5 years), and 3.5 bedrooms or fewer.'
}

if include_property_class:
    classes = active_df[active_df['CBSA_TITLE'] == selected_cbsa]['PROPERTY_CLASS'].unique()
    selected_class = st.selectbox("Select Property Class", sorted(classes))
    
    # Expander for property class descriptions
    with st.expander("Property Class Descriptions", expanded=False):
        for class_num, description in property_class_descriptions.items():
            st.markdown(f"**Class {class_num}**")
            st.write(description)
            st.write("---")
    
    # One-line info for selected class
    selected_desc = property_class_descriptions.get(str(selected_class), "No description available.")
    st.info(f"Currently displaying: {selected_class} — {selected_desc}")

# Year range filtering
st.subheader("Time Range Filter")
df_temp = active_df[active_df['CBSA_TITLE'] == selected_cbsa]
if include_property_class:
    df_temp = df_temp[df_temp['PROPERTY_CLASS'] == selected_class]

min_year = df_temp['MONTH_YEAR'].dt.year.min()
max_year = df_temp['MONTH_YEAR'].dt.year.max()


start_year, end_year = st.slider("Select Year Range", 
                                min_value=int(min_year),
                                max_value=int(max_year), 
                                value=(2020, 2025))

# Filter data
df = active_df[active_df['CBSA_TITLE'] == selected_cbsa]
if include_property_class:
    df = df[df['PROPERTY_CLASS'] == selected_class]

# Apply year filter
df = df[(df['MONTH_YEAR'].dt.year >= start_year) & (df['MONTH_YEAR'].dt.year <= end_year)]
df = df.sort_values('MONTH_YEAR')

st.success(f"Data loaded: {len(df)} observations for {selected_cbsa}")
st.info(f"Date range: {start_year} - {end_year}")

# Step 2: Dual-Axis Analysis
st.header("Step 2: Dual-Axis Analysis")

with st.expander("Dual-Axis Analysis", expanded=True):
    available_metrics = [
        'PCT_LISTINGS_CLOSED',
        'AVG_DAYS_ON_MARKET', 'MEDIAN_CLOSE_PRICE',
        'AVG_CLOSE_TO_LIST_RATIO', 'PCT_CHANGE_IN_LISTINGS'
    ]
    available_in_data = [metric for metric in available_metrics if metric in df.columns]

    if len(available_in_data) >= 1:
        col1, col2 = st.columns(2)
        with col1:
            primary_metric = st.selectbox("Select Primary Metric (Left Y-axis)", available_in_data, 
                                            index=available_in_data.index('MEDIAN_CLOSE_PRICE') if 'MEDIAN_CLOSE_PRICE' in available_in_data else 0,
                                            key="primary_metric")
        with col2:
            use_secondary = st.checkbox("Use Secondary Axis", value=True, key="use_secondary")
            if use_secondary:
                remaining_metrics = [m for m in available_in_data if m != primary_metric]
                if remaining_metrics:
                    secondary_metric = st.selectbox("Select Secondary Metric (Right Y-axis)", remaining_metrics, 
                                                    index=0, key="secondary_metric")
                else:
                    use_secondary = False
                    st.warning("No other metrics available for secondary axis")
        
        if primary_metric in df.columns and len(df) >= 2:
            # Create plot with matplotlib
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Primary metric (left y-axis)
            color1 = 'blue'
            ax1.set_xlabel('Month/Year')
            ax1.set_ylabel(primary_metric, color=color1)
            line1 = ax1.plot(df['MONTH_YEAR'], df[primary_metric], color=color1, linewidth=2, label=primary_metric)
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Secondary metric (right y-axis) - optional
            if use_secondary and secondary_metric in df.columns:
                ax2 = ax1.twinx()
                color2 = 'red'
                ax2.set_ylabel(secondary_metric, color=color2)
                line2 = ax2.plot(df['MONTH_YEAR'], df[secondary_metric], color=color2, linewidth=2, label=secondary_metric)
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Add legend for both metrics
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Correlation analysis for dual-axis
                correlation = df[primary_metric].corr(df[secondary_metric])
                st.write(f"**Correlation between {primary_metric} and {secondary_metric}:** {correlation:.3f}")
                
                if correlation < -0.3:
                    st.info(f"Strong negative correlation: Higher {primary_metric} tends to correspond to lower {secondary_metric}")
                elif correlation > 0.3:
                    st.info(f"Strong positive correlation: Higher {primary_metric} tends to correspond to higher {secondary_metric}")
                else:
                    st.info(f"Weak correlation: {primary_metric} and {secondary_metric} are not strongly related")
                
                plt.title(f'Dual-Axis Analysis: {primary_metric} vs {secondary_metric}')
            else:
                # Single axis plot
                ax1.legend(loc='upper left')
                plt.title(f'Single-Axis Analysis: {primary_metric}')
            
            # Enable zoom and pan functionality
            from matplotlib.widgets import Button
            import matplotlib.dates as mdates
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add grid for better readability
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Display the plot with zoom functionality
            st.pyplot(fig, use_container_width=True)
            
    else:
        st.error("Need at least 1 metric available for analysis.")

# Step 3: Individual Metric Analysis
st.header("Step 3: Individual Metric Analysis")
st.markdown("""
This section provides detailed analysis of individual metrics including:
- Time series visualization
- Summary statistics 
- Stationarity tests
- Autocorrelation analysis
- Differencing analysis
""")

available_metrics = [col for col in df.columns if col in [
    'PCT_LISTINGS_CLOSED',
    'AVG_DAYS_ON_MARKET', 'MEDIAN_CLOSE_PRICE', 
    'AVG_CLOSE_TO_LIST_RATIO', 'PCT_CHANGE_IN_LISTINGS'
]]

# Create dropdown for metric selection
selected_metric = st.selectbox("Select Metric for Analysis", 
                             available_metrics,
                             index=available_metrics.index('MEDIAN_CLOSE_PRICE'))

if selected_metric:
    with st.expander(f"{selected_metric} Analysis", expanded=False):
        # Time series plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['MONTH_YEAR'], df[selected_metric], linewidth=2)
        ax.set_title(f"{selected_metric} Over Time")
        ax.set_xlabel("Month/Year")
        ax.set_ylabel(selected_metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{df[selected_metric].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[selected_metric].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[selected_metric].std():.2f}")
        
        # Stationarity Tests
        st.subheader("Stationarity Analysis")
        series = df[selected_metric].dropna()
        
        if len(series) > 10:  # Need sufficient data for tests
            # ADF Test
            adf_result = adfuller(series)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Augmented Dickey-Fuller Test:**")
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"p-value: {adf_result[1]:.4f}")
                st.write(f"Critical Values:")
                for key, value in adf_result[4].items():
                    st.write(f"  {key}: {value:.3f}")
                
                if adf_result[1] < 0.05:
                    st.success("✅ Series is stationary (reject null hypothesis)")
                else:
                    st.error("❌ Series is non-stationary (fail to reject null hypothesis)")
            
            # KPSS Test
            try:
                kpss_result = kpss(series, regression='c')
                with col2:
                    st.write("**KPSS Test:**")
                    st.write(f"KPSS Statistic: {kpss_result[0]:.4f}")
                    st.write(f"p-value: {kpss_result[1]:.4f}")
                    st.write(f"Critical Values:")
                    for key, value in kpss_result[3].items():
                        st.write(f"  {key}: {value:.3f}")
                    
                    if kpss_result[1] > 0.05:
                        st.success("✅ Series is stationary (fail to reject null hypothesis)")
                    else:
                        st.error("❌ Series is non-stationary (reject null hypothesis)")
            except:
                st.warning("KPSS test could not be performed")
        else:
            st.warning("Insufficient data for stationarity tests")
        
        # ACF and PACF Plots
        st.subheader("Autocorrelation Analysis")
        if len(series) > 10:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            plot_acf(series, ax=ax1, lags=min(20, len(series)//2))
            ax1.set_title(f'Autocorrelation Function (ACF) - {selected_metric}')
            
            plot_pacf(series, ax=ax2, lags=min(20, len(series)//2))
            ax2.set_title(f'Partial Autocorrelation Function (PACF) - {selected_metric}')
            
            plt.tight_layout()
            st.pyplot(fig)
            

        else:
            st.warning("Insufficient data for ACF/PACF analysis")
        
        # Differencing Analysis
        st.subheader("Differencing Analysis")
        if len(series) > 2:
            first_diff = series.diff().dropna()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Series**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(series.index, series.values)
                ax.set_title(f"Original {selected_metric}")
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(series) > 10:
                    adf_orig = adfuller(series)
                    st.write(f"ADF p-value: {adf_orig[1]:.4f}")
            
            with col2:
                st.write("**First Difference**")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(first_diff.index, first_diff.values)
                ax.set_title(f"First Difference of {selected_metric}")
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(first_diff) > 10:
                    adf_diff = adfuller(first_diff)
                    st.write(f"ADF p-value: {adf_diff[1]:.4f}")
                    if adf_diff[1] < 0.05:
                        st.success("✅ First difference is stationary")
                    else:
                        st.warning("⚠️ First difference may still be non-stationary")
            
            if len(series) > 12:
                st.write("**Seasonal Differencing (12-month)**")
                seasonal_diff = series.diff(12).dropna()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(seasonal_diff.index, seasonal_diff.values)
                ax.set_title(f"Seasonal Difference (12-month) of {selected_metric}")
                plt.tight_layout()
                st.pyplot(fig)
                
                if len(seasonal_diff) > 10:
                    adf_seasonal = adfuller(seasonal_diff)
                    st.write(f"ADF p-value: {adf_seasonal[1]:.4f}")
                    if adf_seasonal[1] < 0.05:
                        st.success("✅ Seasonal differencing achieves stationarity")
                        st.info("💡 Consider using seasonal differencing in your model")
                    else:
                        st.warning("⚠️ Seasonal differencing may not be necessary")
                        st.info("💡 Consider using regular differencing instead")
        else:
            st.warning("Insufficient data for differencing analysis")

# Step 4: VAR Analysis

def check_stationarity(df):
    """
    Return a dictionary of {col: p-value} for the ADF test on each column of the DataFrame.
    """
    return {col: adfuller(df[col].dropna())[1] for col in df.columns}

def all_stationary(pvals, alpha=0.05):
    """
    Return True if all p-values are below the significance level (default 0.05), else False.
    """
    return all(p < alpha for p in pvals.values())

st.header("Step 4: VAR Analysis")

available_metrics = [
    'PCT_LISTINGS_CLOSED',
    'AVG_DAYS_ON_MARKET', 'MEDIAN_CLOSE_PRICE',
    'AVG_CLOSE_TO_LIST_RATIO', 'PCT_CHANGE_IN_LISTINGS'
]
selected_metrics = st.multiselect(
    "Select Metrics for VAR Model (2+ required)", 
    available_metrics, 
    default=['MEDIAN_CLOSE_PRICE', 'AVG_DAYS_ON_MARKET']
)

if st.button("Run VAR Analysis"):
    if len(selected_metrics) >= 2:
        st.write(f"**Selected metrics for VAR model:** {', '.join(selected_metrics)}")
        df_var = df.set_index('MONTH_YEAR')[selected_metrics].dropna()
        differencing_applied = "None"
        
        # 1. Check stationarity
        pvals = check_stationarity(df_var)
        if not all_stationary(pvals):
            st.warning("Some series are non-stationary. Applying first differencing.")
            df_var = df_var.diff().dropna()
            differencing_applied = "First difference"
            # Check again
            pvals = check_stationarity(df_var)
            if not all_stationary(pvals):
                st.warning("Some series are still non-stationary after first differencing. Applying seasonal differencing (12).")
                df_var = df_var.diff(12).dropna()
                differencing_applied = "First + Seasonal (12)"
                # Final check
                pvals = check_stationarity(df_var)
                if not all_stationary(pvals):
                    st.error("Warning: Some series remain non-stationary even after differencing. Results may not be reliable.")
        else:
            st.success("All series are stationary. No differencing applied.")

        st.info(f"Differencing applied for VAR: **{differencing_applied}**")
        st.write("ADF p-values after differencing:", pvals)

        n_obs = df_var.shape[0]

        if n_obs >= 24:
            split = int(n_obs * 0.9)
            train, test = df_var.iloc[:split], df_var.iloc[split:]
            
            # Grid search over lag orders
            st.subheader("VAR Lag Selection Grid Search")
            max_lag = min(12, len(train) // 3)  # Don't exceed 1/3 of data
            
            results = []
            for p in range(1, max_lag + 1):
                try:
                    model = VAR(train)
                    fitted_model = model.fit(p)
                    
                    # Calculate metrics
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    # AICc (small sample correction)
                    n = len(train)
                    k = p * len(selected_metrics)**2 + len(selected_metrics)
                    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
                    
                    # SSR (Sum of Squared Residuals for forecast)
                    forecast = fitted_model.forecast(train.values[-p:], steps=len(test))
                    forecast_df = pd.DataFrame(forecast, columns=train.columns, index=test.index)
                    residuals = test - forecast_df
                    ssr = (residuals ** 2).sum().sum()
                    
                    results.append({
                        'p': p,
                        'AIC': aic,
                        'AICc': aicc,
                        'BIC': bic,
                        'SSR': ssr
                    })
                except:
                    continue
            
            if results:
                results_df = pd.DataFrame(results)
                
                # Display results table
                st.write("**Lag Selection Results:**")
                
                # Find optimal lags
                optimal_aic = results_df.loc[results_df['AIC'].idxmin(), 'p']
                optimal_aicc = results_df.loc[results_df['AICc'].idxmin(), 'p']
                optimal_bic = results_df.loc[results_df['BIC'].idxmin(), 'p']
                optimal_ssr = results_df.loc[results_df['SSR'].idxmin(), 'p']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Optimal AIC Lag", optimal_aic)
                with col2:
                    st.metric("Optimal AICc Lag", optimal_aicc)
                with col3:
                    st.metric("Optimal BIC Lag", optimal_bic)
                with col4:
                    st.metric("Optimal SSR Lag", optimal_ssr)
                
                # Plot selection criteria
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                ax1.plot(results_df['p'], results_df['AIC'], 'o-', color='blue')
                ax1.set_title('AIC vs Lag Order')
                ax1.set_xlabel('Lag Order (p)')
                ax1.set_ylabel('AIC')
                ax1.axvline(optimal_aic, color='red', linestyle='--', alpha=0.7)
                
                ax2.plot(results_df['p'], results_df['AICc'], 'o-', color='green')
                ax2.set_title('AICc vs Lag Order')
                ax2.set_xlabel('Lag Order (p)')
                ax2.set_ylabel('AICc')
                ax2.axvline(optimal_aicc, color='red', linestyle='--', alpha=0.7)
                
                ax3.plot(results_df['p'], results_df['BIC'], 'o-', color='orange')
                ax3.set_title('BIC vs Lag Order')
                ax3.set_xlabel('Lag Order (p)')
                ax3.set_ylabel('BIC')
                ax3.axvline(optimal_bic, color='red', linestyle='--', alpha=0.7)
                
                ax4.plot(results_df['p'], results_df['SSR'], 'o-', color='purple')
                ax4.set_title('SSR vs Lag Order')
                ax4.set_xlabel('Lag Order (p)')
                ax4.set_ylabel('SSR')
                ax4.axvline(optimal_ssr, color='red', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Use optimal lag for final model
                st.subheader("Final VAR Model with Optimal Lag")
                optimal_lag = optimal_bic  # BIC is often preferred for lag selection
                
                model = VAR(train)
                results = model.fit(optimal_lag)

                forecast = results.forecast(train.values[-optimal_lag:], steps=len(test))
                forecast_df = pd.DataFrame(forecast, columns=train.columns, index=test.index)

                st.subheader("Forecast vs. Actual")
                for i, col in enumerate(train.columns):
                    # First plot: Actual values and forecasts
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.plot(train.index, train[col], label='Train', linewidth=2, alpha=0.8)
                    ax1.plot(test.index, test[col], label='Actual', marker='o', markersize=6, linestyle='-', linewidth=2, alpha=0.8)
                    
                    # Re-cumulate forecast to original scale if differencing was applied
                    if differencing_applied != "None":
                        last_train_value = train[col].iloc[-1]
                        recum_forecast = forecast_df[col].cumsum() + last_train_value
                        ax1.plot(forecast_df.index, recum_forecast, label='Forecast', linestyle='--', linewidth=2, color='tab:orange')
                        
                        # Confidence intervals for recumulated forecast
                        try:
                            fcst, lower, upper = results.forecast_interval(train.values[-optimal_lag:], steps=len(test), alpha=0.05)
                            lower_recum = lower[:, i].cumsum() + last_train_value
                            upper_recum = upper[:, i].cumsum() + last_train_value
                            ax1.fill_between(forecast_df.index, lower_recum, upper_recum, color='gray', alpha=0.15, label='95% CI')
                        except Exception:
                            stderr = residuals[col].std()
                            ci_upper = recum_forecast + 1.96 * stderr
                            ci_lower = recum_forecast - 1.96 * stderr
                            ax1.fill_between(forecast_df.index, ci_lower, ci_upper, color='gray', alpha=0.15, label='Approx. 95% CI')
                    else:
                        ax1.plot(forecast_df.index, forecast_df[col], label='Forecast', linestyle='--', linewidth=2, color='tab:orange')
                        
                        # Confidence intervals for non-differenced forecast
                        try:
                            fcst, lower, upper = results.forecast_interval(train.values[-optimal_lag:], steps=len(test), alpha=0.05)
                            ax1.fill_between(forecast_df.index, lower[:, i], upper[:, i], color='gray', alpha=0.15, label='95% CI')
                        except Exception:
                            stderr = residuals[col].std()
                            ci_upper = forecast_df[col] + 1.96 * stderr
                            ci_lower = forecast_df[col] - 1.96 * stderr
                            ax1.fill_between(forecast_df.index, ci_lower, ci_upper, color='gray', alpha=0.15, label='Approx. 95% CI')

                    ax1.set_title(f"{col} - Forecast vs Actual", fontsize=12, pad=15)
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='best', framealpha=0.9)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    # Second plot: Differences/Residuals
                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                    if differencing_applied != "None":
                        differences = test[col] - recum_forecast
                    else:
                        differences = test[col] - forecast_df[col]
                    ax2.plot(forecast_df.index, differences, color='tab:red', marker='o', label='Actual - Forecast')
                    ax2.axhline(0, linestyle='--', color='gray')
                    ax2.set_title(f"{col} - Forecast Error", fontsize=12, pad=15)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='best', framealpha=0.9)
                    plt.tight_layout()
                    st.pyplot(fig2)

                st.subheader("Residuals")
                residuals = test - forecast_df
                for col in residuals.columns:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(residuals.index, residuals[col], marker='o')
                    ax.axhline(0, linestyle='--', color='gray')
                    ax.set_title(f"Residuals – {col}")
                    plt.tight_layout()
                    st.pyplot(fig)

                st.markdown("### Forecast RMSE (Root Mean Squared Error)")
                st.markdown("""
                - **In-sample RMSE** measures how well the model fits the training data (lower is better).
                - **Out-of-sample RMSE** measures how well the model predicts new, unseen data (lower is better).
                - A large gap between in-sample and out-of-sample RMSE may indicate overfitting.
                """)

                for col in forecast_df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        # In-sample
                        in_sample_pred = results.fittedvalues[col]
                        in_sample_actual = train[col].iloc[results.k_ar:]
                        in_sample_rmse = rmse(in_sample_actual, in_sample_pred)
                        in_sample_mape = mape(in_sample_actual, in_sample_pred)
                        st.write(f"**{col} In-Sample:**")
                        st.write(f"- RMSE: {in_sample_rmse:.2f}")
                        st.write(f"- MAPE: {in_sample_mape:.2f}%")
                    
                    with col2:
                        # Out-of-sample
                        out_sample_rmse = rmse(test[col], forecast_df[col])
                        out_sample_mape = mape(test[col], forecast_df[col])
                        st.write(f"**{col} Out-of-Sample:**") 
                        st.write(f"- RMSE: {out_sample_rmse:.2f}")
                        st.write(f"- MAPE: {out_sample_mape:.2f}%")
                    st.write('---')

                # Calculate residual statistics
                residual_stats = {}
                for col in residuals.columns:
                    stats = {
                        'mean': residuals[col].mean(),
                        'std': residuals[col].std(),
                        'max_abs': abs(residuals[col]).max(),
                        'autocorr': residuals[col].autocorr()
                    }
                    residual_stats[col] = stats

                # Evaluate model fitness
                def evaluate_model_fitness(in_rmse, out_rmse, res_stats):
                    concerns = []
                    strengths = []
                    
                    # Check residual mean (should be close to 0)
                    if abs(res_stats['mean']) > 0.1 * res_stats['std']:
                        concerns.append("systematic bias (residual mean not close to 0)")
                    else:
                        strengths.append("unbiased predictions (residual mean close to 0)")
                        
                    # Check autocorrelation (should be low)
                    if abs(res_stats['autocorr']) > 0.3:
                        concerns.append("significant autocorrelation in residuals")
                    else:
                        strengths.append("no significant autocorrelation")
                        
                    # Compare in-sample vs out-of-sample RMSE
                    rmse_ratio = out_rmse / in_rmse
                    if rmse_ratio > 1.5:
                        concerns.append("possible overfitting (out-of-sample RMSE much larger)")
                    elif rmse_ratio < 1.2:
                        strengths.append("good generalization to new data")
                        
                    return strengths, concerns

                st.markdown("""
                **Interpreting Residuals:**
                - Residuals are the differences between the actual values and the model's predictions.
                - In a well-specified time series model, residuals should resemble white noise: they should have a mean close to zero, constant variance, and no autocorrelation.
                """)

                st.subheader("Model Fitness Analysis")
                
                for col in residuals.columns:
                    st.markdown(f"### Analysis for {col}")
                    
                    # Get statistics
                    stats = residual_stats[col]
                    strengths, concerns = evaluate_model_fitness(
                        in_sample_rmse if 'in_sample_rmse' in locals() else 0,
                        out_sample_rmse if 'out_sample_rmse' in locals() else 0,
                        stats
                    )
                    
                    st.markdown(f"""
                    **Residual Statistics:**
                    - Mean: {stats['mean']:.4f}
                    - Standard Deviation: {stats['std']:.4f}
                    - Maximum Absolute Error: {stats['max_abs']:.4f}
                    - First-order Autocorrelation: {stats['autocorr']:.4f}
                    """)
                    
                    if strengths:
                        st.success("**Model Strengths:**\n" + "\n".join(f"- {s}" for s in strengths))
                    
                    if concerns:
                        st.warning("**Areas for Improvement:**\n" + "\n".join(f"- {c}" for c in concerns))
                    
                    # Overall assessment
                    if len(concerns) == 0:
                        st.success("✅ This model shows excellent predictive power")
                    elif len(concerns) <= 1:
                        st.info("ℹ️ This model shows good predictive power with minor areas for improvement")
                    else:
                        st.warning("⚠️ This model might benefit from further refinement")
                    
                    st.markdown("---")

                st.session_state['var_results'] = results
                st.session_state['var_optimal_lag'] = optimal_lag
                st.session_state['var_selected_vars'] = selected_metrics
            else:
                st.error("Could not fit VAR models with any lag order")
        else:
            st.warning(f"Not enough data points for VAR model (need >= 24 rows, have {n_obs}).")
    else:
        st.warning("Please select at least two metrics for VAR modeling.")

# Reset button for Step 4
if st.session_state.get('var_results', None) is not None:
    if st.button("Reset VAR Analysis"):
        st.session_state['var_results'] = None
        st.session_state['var_optimal_lag'] = None
        st.session_state['var_selected_vars'] = None

# Step 5: Impulse Response & Variance Decomposition (VAR)
st.header("Step 5: Impulse Response & Variance Decomposition (VAR)")

def interpret_irf(irf_obj, variables, horizon=12, threshold_ratio=0.05):
    """
    Generate human-readable interpretations for each IRF panel.
    - irf_obj: statsmodels IRAnalysis object
    - variables: list of variable names (in order)
    - horizon: number of periods ahead
    - threshold_ratio: fraction of max(abs(response)) to consider as 'significant'
    """
    n = len(variables)
    text_blocks = []
    irf_vals = irf_obj.irfs[:horizon+1]  # shape: (horizon+1, n, n)
    for i, shock_var in enumerate(variables):
        for j, response_var in enumerate(variables):
            response = irf_vals[:, j, i]
            max_abs = np.max(np.abs(response))
            threshold = threshold_ratio * max_abs if max_abs > 0 else 0.0
            start_val = response[0]
            end_val = response[-1]
            direction = "positive" if start_val > 0 else "negative" if start_val < 0 else "no"
            decay = "decays" if np.abs(end_val) < np.abs(start_val) else "persists"
            significance = "statistically significant" if max_abs > threshold else "likely minor or insignificant"
            # Panel label
            panel = f"{shock_var} → {response_var}"
            # Human-readable summary
            if i == j:
                summary = (
                    f"**Self-shock:** A shock to **{shock_var}** has a {direction} effect on itself that {decay} over {horizon} periods. "
                    f"The effect is {significance}."
                )
            else:
                summary = (
                    f"**Cross-shock:** A shock to **{shock_var}** has a {direction} effect on **{response_var}** that {decay} over {horizon} periods. "
                    f"The effect is {significance}."
                )
            # Economic context (optional, can be expanded)
            if direction == "positive" and decay == "decays":
                econ = "A positive shock leads to an increase that gradually returns to normal."
            elif direction == "negative" and decay == "decays":
                econ = "A negative shock leads to a decrease that gradually returns to normal."
            elif decay == "persists":
                econ = "The effect is persistent, indicating lasting impact."
            else:
                econ = "No substantial effect detected."
            text_blocks.append(f"**{panel}:**\n{summary}\n_{econ}_\n")
    return text_blocks

def interpret_fevd(fevd_obj, variables, horizon=8, threshold=0.1):
    """
    Generate concise interpretations for FEVD results.
    - fevd_obj: statsmodels FEVD object
    - variables: list of variable names (in order)
    - horizon: number of periods to interpret
    - threshold: minimum fraction to consider as 'notable'
    """
    text_blocks = []
    for i, var in enumerate(variables):
        contribs = fevd_obj.decomp[:horizon+1, i, :]
        own = contribs[:, i]
        others = [contribs[:, j] for j in range(len(variables)) if j != i]
        max_other = np.max(others) if others else 0
        main_other_idx = np.argmax([np.max(c) for c in others]) if others else None
        main_other = variables[main_other_idx+1] if main_other_idx is not None else None

        if np.all(own > 0.9):
            summary = f"Most of the forecast error variance in **{var}** is explained by its own shocks."
        elif max_other > threshold:
            summary = f"While most of the forecast error variance in **{var}** is due to its own shocks, {main_other} also contributes noticeably over time."
        else:
            summary = f"**{var}**'s forecast error variance is mainly self-driven, with only minor influence from other variables."
        text_blocks.append(summary)
    return text_blocks
if st.session_state.get('var_results', None) is not None:
    results = st.session_state['var_results']
    optimal_lag = st.session_state['var_optimal_lag']
    selected_vars = st.session_state['var_selected_vars']
    horizon = 12  # Hardcoded horizon value

    st.info(f"Using lag order: {optimal_lag} and variables: {', '.join(selected_vars)} from Step 4.")

    # IRF
    st.subheader("Impulse Response Functions (IRF)")
    fig_irf = results.irf(horizon).plot(orth=False)
    plt.suptitle("Impulse Response Functions", fontsize=14)
    st.pyplot(fig_irf.figure)
    st.markdown(
        "The IRF shows the effect of a one-time shock to one variable on the current and future values of all variables in the system."
    )

    # FEVD
    st.subheader("Forecast Error Variance Decomposition (FEVD)")
    fig_fevd = results.fevd(horizon).plot()
    plt.suptitle("Forecast Error Variance Decomposition", fontsize=14)
    st.pyplot(fig_fevd.figure)
    st.markdown(
        "FEVD shows the proportion of the forecast error variance of each variable that can be attributed to shocks in each variable in the system."
    )

    # IRF Interpretation
    st.markdown("#### IRF Interpretation")
    irf_obj = results.irf(horizon)
    interpretations = interpret_irf(irf_obj, selected_vars, horizon=horizon)
    for interp in interpretations:
        st.markdown(interp)

    # FEVD Interpretation
    st.markdown("#### FEVD Interpretation")
    fevd_obj = results.fevd(horizon)
    interpretations = interpret_fevd(fevd_obj, selected_vars, horizon=horizon)
    for interp in interpretations:
        st.markdown(f"- {interp}")
else:
    st.info("Please complete Step 4 (VAR Analysis) before running Step 5.")

