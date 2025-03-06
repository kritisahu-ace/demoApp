import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
import os
import altair as alt
# import streamlit as st

def load_data():
    # Load dataset (replace with your dataset)
    # Construct the file path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "used_cars_india.csv")
    # file_path = "used_cars_india.csv"

    df = pd.read_csv(file_path, sep="\t")
    df['Sales Date'] = pd.to_datetime(df['Sales Date'])
    return df

def forecast_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    try:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)  # Forecast for the next quarter
        return forecast[0]
    except:
        return 0  # Return 0 if model fitting fails

def country_analysis():
    # df = load_data()
    # df_agg = df.groupby(['Model', 'Country', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')
    # df_pivot = df_agg.pivot(index='Sales Date', columns=['Model', 'Country'], values='Sales').fillna(0)

    # results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    # forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
    #                            columns=['Model', 'Country', 'Forecasted Sales'])

    # top_models = forecast_df.nlargest(10, 'Forecasted Sales')
    # top_models.sort_values('Forecasted Sales', inplace = True, ascending=False)
    # top_models['Model By Country'] = top_models['Model'] + " (" + top_models['Country'] + ")"
    top_models= pd.read_csv("top_models.csv")
    chart = (
        alt.Chart(top_models)
        .mark_bar()
        .encode(
            alt.X("Model By Country:N", axis=alt.Axis(labelAngle=-45),sort=top_models['Forecasted Sales']),
            alt.Y("Forecasted Sales"),
            alt.Color("Model By Country:N"),
            alt.Tooltip(["Model By Country", "Forecasted Sales"]),
        )
        .interactive()
    )
    # st.altair_chart(chart)
    # Plot
    # plt.figure(figsize=(12, 6))
    # bars = plt.barh(top_models['Model'] + " (" + top_models['Country'] + ")", top_models['Forecasted_Sales'], color='skyblue')
    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.0f}', ha='left', va='center')
    # plt.xlabel('Forecasted Sales (Next Quarter)')
    # plt.ylabel('Model (Country)')
    # plt.title('Top 10 Selling Car Models for Next Quarter by Country')
    # plt.gca().invert_yaxis()
    return chart

if __name__ == "__main__":
    country_analysis().show()
