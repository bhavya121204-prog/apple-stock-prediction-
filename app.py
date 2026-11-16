
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Page configuration
st.set_page_config(
    page_title="Apple Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("🍎 Apple Stock Price Prediction")
st.write("30 Trading Days Stock Price Forecasting using ARIMA")

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Apple Stock CSV",
    type=["csv"]
)

# Load dataset
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["Close"] = pd.to_numeric(
        data["Close"], errors="coerce"
    )

    data = data.dropna(subset=["Date", "Close"])

    return data

try:
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        data = load_data("P721 DATASET.csv")

    if len(data) < 40:
        st.error("Please upload at least 40 valid stock records.")
        st.stop()

    # Dataset overview
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Start Date", data["Date"].min().strftime("%Y-%m-%d"))
    col3.metric("End Date", data["Date"].max().strftime("%Y-%m-%d"))

    st.dataframe(data.tail(10), use_container_width=True)

    # Historical stock prices
    st.subheader("📈 Historical Closing Prices")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data["Date"],
        data["Close"],
        label="Closing Price"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.set_title("Apple Historical Stock Prices")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Descriptive statistics
    st.subheader("📋 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

    # ARIMA Model
    st.subheader("🤖 ARIMA Model Prediction")

    st.write("Selected Model: ARIMA (1,1,3)")

    if st.button("Generate 30-Day Forecast"):

        with st.spinner("Training ARIMA model..."):

            close_prices = data["Close"]

            model = ARIMA(
                close_prices,
                order=(1, 1, 3)
            )

            fitted_model = model.fit()

            # Forecast next 30 trading days
            forecast_result = fitted_model.get_forecast(
                steps=30
            )

            predictions = forecast_result.predicted_mean
            confidence = forecast_result.conf_int()

            # Generate future business dates
            forecast_dates = pd.bdate_range(
                start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
                periods=30
            )

            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted_Close": predictions.values,
                "Lower_CI": confidence.iloc[:, 0].values,
                "Upper_CI": confidence.iloc[:, 1].values
            })

            # Display forecast
            st.success("Forecast generated successfully!")

            st.subheader("📅 Next 30 Trading Days Prediction")

            st.dataframe(
                forecast_df.round(2),
                use_container_width=True
            )

            # Forecast graph
            st.subheader("📈 Actual vs Predicted Stock Prices")

            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(
                data["Date"].iloc[-90:],
                data["Close"].iloc[-90:],
                label="Historical Price"
            )

            ax.plot(
                forecast_df["Date"],
                forecast_df["Predicted_Close"],
                linestyle="--",
                label="Predicted Price"
            )

            ax.fill_between(
                forecast_df["Date"],
                forecast_df["Lower_CI"],
                forecast_df["Upper_CI"],
                alpha=0.2,
                label="95% Confidence Interval"
            )

            ax.set_title("Apple Stock Price - 30 Day Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Closing Price (USD)")
            ax.legend()
            ax.grid(True)

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # Download predictions
            csv = forecast_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="apple_stock_forecast.csv",
                mime="text/csv"
            )

            # Model summary
            with st.expander("View ARIMA Model Summary"):
                st.text(str(fitted_model.summary()))

except FileNotFoundError:
    st.error(
        "Dataset not found. Upload your Apple stock CSV file."
    )

except Exception as e:
    st.error(f"Error: {e}")

st.sidebar.info(
    "Forecasting Model: ARIMA (1,1,3)"
)

st.markdown("---")
st.caption("Apple Stock Price Prediction | Machine Learning Project")
