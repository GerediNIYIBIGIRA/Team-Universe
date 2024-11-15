# graph.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Initialize dataset
youth_labour_df = pd.read_csv('youth_data_up_to_35_years_old.csv')

def plot_youth_unemployment_by_age(youth_labour_df):
    # Calculate average unemployment rate by age group
    average_unemployment = youth_labour_df.groupby('age5')['target_employed16'].mean()
    average_unemployment = 1 - average_unemployment  # Unemployment rate

    # Create the figure with a bar and line plot
    fig = go.Figure()

    # Add the bar plot for unemployment rates
    fig.add_trace(go.Bar(
        x=average_unemployment.index,
        y=average_unemployment,
        name="Unemployment Rate",
        marker_color='salmon'
    ))

    # Add the line plot on top of the bars
    fig.add_trace(go.Scatter(
        x=average_unemployment.index,
        y=average_unemployment,
        mode='lines+markers+text',
        line=dict(color='red', width=2),
        name="Average Unemployment Rate",
        text=[f'{v:.2%}' for v in average_unemployment],
        textposition="top center"  # This replaces 'outside' to avoid the error
    ))

    # Customize layout
    fig.update_layout(
        title='Youth Unemployment Rate by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Average Unemployment Rate',
        yaxis=dict(tickformat=".0%", range=[0, max(average_unemployment) * 1.1]),
        bargap=0.2,
        legend=dict(x=0.85, y=0.95),
        template="simple_white"
    )

    return fig
