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

def forecast_sarima(series, order=(1, 0, 0)):
    """
    Simplified SARIMA forecasting with fewer parameters.
    """
    try:
        model = SARIMAX(series, order=order, seasonal_order=(0, 0, 0, 0))  # No seasonality
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=1)  # Forecast for the next quarter
        return forecast[0]
    except:
        return 0  # Return 0 if model fitting fails

def customer_country_analysis():
    df = load_data()
    
    # Filter customers who change cars frequently (e.g., customers with at least 2 purchases)
    customer_purchase_count = df.groupby(['Customer', 'Country']).size().reset_index(name='Purchase Count')
    frequent_customers = customer_purchase_count[customer_purchase_count['Purchase Count'] >= 2]
    
    # Filter the dataset to include only frequent customers
    df_frequent = df.merge(frequent_customers, on=['Customer', 'Country'])
    
    # Aggregate sales data by Customer, Country, and Sales Date (quarterly)
    df_agg = df_frequent.groupby(['Customer', 'Country', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')
    
    # Pivot the data for time series analysis
    df_pivot = df_agg.pivot(index='Sales Date', columns=['Customer', 'Country'], values='Sales').fillna(0)

    # Parallelize SARIMA forecasting
    results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
                               columns=['Customer', 'Country', 'Forecasted Sales'])

    # Convert 'Forecasted Sales' to numeric type
    forecast_df['Forecasted Sales'] = pd.to_numeric(forecast_df['Forecasted Sales'], errors='coerce')

    # Get top 3 customers for each country
    top_customers_by_country = forecast_df.groupby('Country').apply(lambda x: x.nlargest(3, 'Forecasted Sales')).reset_index(drop=True)

    # top_models = forecast_df.nlargest(10, 'Forecasted Sales')
    top_customers_by_country.sort_values('Forecasted Sales', inplace = True, ascending=False)
    top_customers_by_country['Customer By Country'] = top_customers_by_country['Customer'] + " (" + top_customers_by_country['Country'] + ")"
    # top_customers_by_country.to_csv("try.csv")
    chart = (
        alt.Chart(top_customers_by_country)
        .mark_bar()
        .encode(
            alt.X("Customer By Country:N", axis=alt.Axis(labelAngle=-45),sort=top_customers_by_country['Forecasted Sales']),
            alt.Y("Forecasted Sales"),
            alt.Color("Customer By Country:N"),
            alt.Tooltip(["Customer By Country", "Forecasted Sales"]),
        )
        .interactive()
    )

    return chart

    # # Plot
    # countries = top_customers_by_country['Country'].unique()
    # customers = top_customers_by_country['Customer'].unique()
    # sales_data = {country: top_customers_by_country[top_customers_by_country['Country'] == country]['Forecasted Sales'].values for country in countries}

    # x = np.arange(len(countries))
    # width = 0.2
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for i, customer in enumerate(customers):
    #     ax.bar(x + i * width, [sales_data[country][i] if i < len(sales_data[country]) else 0 for country in countries], width, label=customer)

    # ax.set_xlabel('Country')
    # ax.set_ylabel('Forecasted Car Changes (Next Quarter)')
    # ax.set_title('Top 3 Customers Who Change Cars Frequently for Next Quarter by Country')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(countries, rotation=45, ha='right')
    # ax.legend(title='Customer', bbox_to_anchor=(1.05, 1), loc='upper left')

    # return plt

if __name__ == "__main__":
    customer_country_analysis().show()