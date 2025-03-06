import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
import os
import altair as alt

def load_data():
    # Construct the file path dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "used_cars_india.csv")
    
    # Load dataset
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

def sales_type_district_analysis():
    df = load_data()
    
    # Aggregate sales data by Sales Type, District, and Sales Date (quarterly)
    df_agg = df.groupby(['Sales Type', 'District', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')
    
    # Filter major districts (e.g., districts with at least 1000 total sales)
    district_sales = df_agg.groupby('District')['Sales'].sum().reset_index()
    major_districts = district_sales[district_sales['Sales'] >= 400]['District'].tolist()
    df_agg = df_agg[df_agg['District'].isin(major_districts)]

    df_pivot = df_agg.pivot(index='Sales Date', columns=['Sales Type', 'District'], values='Sales').fillna(0)

    results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
                               columns=['Sales Type', 'District', 'Forecasted Sales'])

    # Convert 'Forecasted Sales' to numeric type
    forecast_df['Forecasted Sales'] = pd.to_numeric(forecast_df['Forecasted Sales'], errors='coerce')

    # Get top order of preferred sales type for each district
    top_sales_types_by_district = forecast_df.groupby('District').apply(lambda x: x.nlargest(3, 'Forecasted Sales')).reset_index(drop=True)

    chart = (
        alt.Chart(top_sales_types_by_district, width=505, height=200).mark_bar(size=10).encode(
            x='Sales Type:N',#, title='', axis=alt.Axis(labelAngle=45, labelLimit=10), sort=top_models_by_state['Forecasted Sales']),
            # xOffset='State',
            color=alt.Color('Sales Type:N'),
            y='Forecasted Sales:Q',
            facet='District:N'
            # column=alt.Column('State', header=alt.Header(orient='bottom', title='')),
        ).configure_header(labelOrient='bottom',
                    labelPadding = 3).configure_facet(spacing=5
 ))
    return chart

    # # Plot
    # districts = top_sales_types_by_district['District'].unique()
    # sales_types = top_sales_types_by_district['Sales Type'].unique()
    # sales_data = {district: top_sales_types_by_district[top_sales_types_by_district['District'] == district]['Forecasted Sales'].values for district in districts}

    # x = np.arange(len(districts))
    # width = 0.2
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for i, sales_type in enumerate(sales_types):
    #     ax.bar(x + i * width, [sales_data[district][i] if i < len(sales_data[district]) else 0 for district in districts], width, label=sales_type)

    # ax.set_xlabel('District')
    # ax.set_ylabel('Forecasted Sales (Next Quarter)')
    # ax.set_title('Top Order of Preferred Sales Type for Next Quarter by Major Districts')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(districts, rotation=45, ha='right')
    # ax.legend(title='Sales Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # return plt

if __name__ == "__main__":
    sales_type_district_analysis().show()