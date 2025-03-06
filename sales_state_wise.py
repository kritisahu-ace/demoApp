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

def state_analysis():
    df = load_data()
    df_agg = df.groupby(['Model', 'State', pd.Grouper(key='Sales Date', freq='Q')]).size().reset_index(name='Sales')
    df_pivot = df_agg.pivot(index='Sales Date', columns=['Model', 'State'], values='Sales').fillna(0)

    results = Parallel(n_jobs=-1)(delayed(forecast_sarima)(df_pivot[column]) for column in df_pivot.columns)
    forecast_df = pd.DataFrame([(column[0], column[1], result) for column, result in zip(df_pivot.columns, results)],
                               columns=['Model', 'State', 'Forecasted Sales'])

    # Convert 'Forecasted Sales' to numeric type
    forecast_df['Forecasted Sales'] = pd.to_numeric(forecast_df['Forecasted Sales'], errors='coerce')

    # Get top 3 models for each state
    top_models_by_state = forecast_df.groupby('State').apply(lambda x: x.nlargest(3, 'Forecasted Sales')).reset_index(drop=True)

    # top_models_by_state.to_csv("123.csv")
    # Plot
    states = top_models_by_state['State'].unique()
    models = top_models_by_state['Model'].unique()
    sales_data = {state: top_models_by_state[top_models_by_state['State'] == state]['Forecasted Sales'].values for state in states}

    chart = (
        alt.Chart(top_models_by_state, width=505, height=200).mark_bar(size=10).encode(
            x='Model:N',#, title='', axis=alt.Axis(labelAngle=45, labelLimit=10), sort=top_models_by_state['Forecasted Sales']),
            # xOffset='State',
            color=alt.Color('Model:N'),
            y='Forecasted Sales:Q',
            facet='State:N'
            # column=alt.Column('State', header=alt.Header(orient='bottom', title='')),
        ).configure_header(labelOrient='bottom',
                    labelPadding = 3).configure_facet(spacing=5
 ))
    #         alt.X("Model By Country:O", axis=alt.Axis(labelAngle=-45),sort=top_models['Forecasted Sales']),
    #         alt.Y("Forecasted Sales"),
    #         alt.Color("Model By Country:O"),
    #         alt.Tooltip(["Model By Country", "Forecasted Sales"]),
    # )

    # cols = ['Model', 'State', 'Forecasted Sales']
    # lst = []
    # for state in states:
    #     lst.append([top_models_by_state[top_models_by_state['State'] == state]['Model'], 2, top_models_by_state[top_models_by_state['State'] == state]['Forecasted Sales'].values])
    # df = pd.DataFrame(lst, columns=cols)

    # df = pd.DataFrame({
    #     'student': ['A', 'A', 'B', 'B', 'C', 'C'],
    #     'subject': ['Math', 'English', 'Math', 'English', 'Math', 'English'],
    #     'current': [98, 88, 68, 72, 92, 92],
    #     'previous': [92, 94, 71, 71, 88, 84],
    # }).melt(
    #     id_vars=['student', 'subject'],
    #     var_name='score_type',
    #     value_name='score'
    # )
    
    # top_models = forecast_df.nlargest(10, 'Forecasted Sales')
    # top_models.sort_values('Forecasted Sales', inplace = True, ascending=False)
    # top_models['Model By Country'] = top_models['Model'] + " (" + top_models['Country'] + ")"

    # chart = (
    #     alt.Chart(top_models_by_state)
    #     .mark_bar()
    #     .encode(
    #         alt.X("State:O", axis=alt.Axis(labelAngle=-45)),
    #         alt.Y("Forecasted Sales"),
    #         alt.Color("State:O"),
    #         alt.Tooltip(["State", "Forecasted Sales"]),
    #     )
    #     .interactive()
    # )

    # x = np.arange(len(states))
    # width = 0.2
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for i, model in enumerate(models):
    #     ax.bar(x + i * width, [sales_data[state][i] if i < len(sales_data[state]) else 0 for state in states], width, label=model)

    # ax.set_xlabel('State')
    # ax.set_ylabel('Forecasted Sales (Next Quarter)')
    # ax.set_title('Top 3 Selling Car Models for Next Quarter by State')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(states, rotation=45, ha='right')
    # ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # return plt
    return chart

if __name__ == "__main__":
    state_analysis().show()