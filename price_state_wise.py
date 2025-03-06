import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
import os
import altair as alt


def load_data():
    # Load dataset (replace with your dataset)
    # Construct the file path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "used_cars_india.csv")
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

def price_state_analysis():
    df = load_data()
    
    # Define price ranges
    bins = [0, 200000, 500000, 1000000, float('inf')]
    labels = ['0-2L', '2L-5L', '5L-10L', '10L+']
    df['Price Range'] = pd.cut(df['Sales Price'], bins=bins, labels=labels)
    
    # Aggregate sales data by Price Range, State, and Sales Date (quarterly)
    df_agg = df.groupby(['Price Range', 'State', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')
    df_pivot = df_agg.pivot(index='Sales Date', columns=['Price Range', 'State'], values='Sales').fillna(0)

    results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
                               columns=['Price Range', 'State', 'Forecasted Sales'])

    # Convert 'Forecasted Sales' to numeric type
    forecast_df['Forecasted Sales'] = pd.to_numeric(forecast_df['Forecasted Sales'], errors='coerce')

    # Get top 3 price ranges for each state
    top_price_ranges_by_state = forecast_df.groupby('State').apply(lambda x: x.nlargest(3, 'Forecasted Sales')).reset_index(drop=True)

    chart = (
        alt.Chart(top_price_ranges_by_state, width=505, height=200).mark_bar(size=10).encode(
            x='Price Range:N',#, title='', axis=alt.Axis(labelAngle=45, labelLimit=10), sort=top_models_by_state['Forecasted Sales']),
            # xOffset='State',
            color=alt.Color('Price Range:N'),
            y='Forecasted Sales:Q',
            facet='State:N'
            # column=alt.Column('State', header=alt.Header(orient='bottom', title='')),
        ).configure_header(labelOrient='bottom',
                    labelPadding = 3).configure_facet(spacing=5
 ))

    return chart

    # # Plot
    # states = top_price_ranges_by_state['State'].unique()
    # price_ranges = top_price_ranges_by_state['Price Range'].unique()
    # sales_data = {state: top_price_ranges_by_state[top_price_ranges_by_state['State'] == state]['Forecasted Sales'].values for state in states}

    # x = np.arange(len(states))
    # width = 0.2
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for i, price_range in enumerate(price_ranges):
    #     ax.bar(x + i * width, [sales_data[state][i] if i < len(sales_data[state]) else 0 for state in states], width, label=price_range)

    # ax.set_xlabel('State')
    # ax.set_ylabel('Forecasted Sales (Next Quarter)')
    # ax.set_title('Top 3 Preferred Price Ranges for Next Quarter by State')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(states, rotation=45, ha='right')
    # ax.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')

    # return plt

if __name__ == "__main__":
    price_state_analysis().show()