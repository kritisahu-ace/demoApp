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

def district_analysis():
    # df = load_data()
    # df_agg = df.groupby(['Model', 'District', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')  
    # # Filter major districts (e.g., districts with at least 1000 total sales)
    # district_sales = df_agg.groupby('District')['Sales'].sum().reset_index()
    # major_districts = district_sales[district_sales['Sales'] >= 400]['District'].tolist()
    # df_agg = df_agg[df_agg['District'].isin(major_districts)]
    # df_pivot = df_agg.pivot(index='Sales Date', columns=['Model', 'District'], values='Sales').fillna(0)

    # # Filter districts with sufficient data (e.g., at least 10 sales)
    # districts_to_keep = df_pivot.columns[df_pivot.sum() > 10]
    # df_pivot = df_pivot[districts_to_keep]    

    # results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    # forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
    #                            columns=['Model', 'District', 'Forecasted Sales'])

    # # Convert 'Forecasted Sales' to numeric type
    # forecast_df['Forecasted Sales'] = pd.to_numeric(forecast_df['Forecasted Sales'], errors='coerce')

    # # Get top 3 models for each district
    # top_models_by_district = forecast_df.groupby('District').apply(lambda x: x.nlargest(3, 'Forecasted Sales')).reset_index(drop=True)
    top_models_by_district = pd.read_csv("top_models_by_district.csv")
    chart = (
        alt.Chart(top_models_by_district, width=505, height=200).mark_bar(size=10).encode(
            x='Model:N',#, title='', axis=alt.Axis(labelAngle=45, labelLimit=10), sort=top_models_by_state['Forecasted Sales']),
            # xOffset='State',
            color=alt.Color('Model:N'),
            y='Forecasted Sales:Q',
            facet='District:N'
            # column=alt.Column('State', header=alt.Header(orient='bottom', title='')),
        ).configure_header(labelOrient='bottom',
                    labelPadding = 3).configure_facet(spacing=5
 ))
    return chart

    # # Plot
    # districts = top_models_by_district['District'].unique()
    # models = top_models_by_district['Model'].unique()
    # sales_data = {district: top_models_by_district[top_models_by_district['District'] == district]['Forecasted Sales'].values for district in districts}

    # x = np.arange(len(districts))
    # width = 0.2
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for i, model in enumerate(models):
    #     ax.bar(x + i * width, [sales_data[district][i] if i < len(sales_data[district]) else 0 for district in districts], width, label=model)

    # ax.set_xlabel('District')
    # ax.set_ylabel('Forecasted Sales (Next Quarter)')
    # ax.set_title('Top 3 Selling Car Models for Next Quarter by District')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(districts, rotation=45, ha='right')
    # ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # return plt

if __name__ == "__main__":
    district_analysis().show()
