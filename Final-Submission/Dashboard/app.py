from scipy.stats import linregress
import dash_bootstrap_components as dbc
import plotly.tools as tls
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, html, dcc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
from utils.data_processing import load_data, load_data_HDI, load_data_Unemployment
import openai
from config import OPENAI_API_KEY
from dash.exceptions import PreventUpdate
from statsmodels.tsa.arima.model import ARIMA
import os


# Fetch secret keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OpenAI API key is missing. Please set OPENAI_API_KEY in the environment variables.")

openai.api_key = OPENAI_API_KEY

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# external JavaScript files
external_scripts = [
    'http://localhost:8000/copilot/index.js'
]
# Load and cache data for performance
data = load_data()
data1 = load_data_Unemployment()
data2 = load_data_HDI()
label_mapping ={
    0:'Unemployed',
    1: 'Employment'    
}
app = Dash(__name__, external_scripts=external_scripts,external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
])

# Sidebar layout with navigation options
sidebar = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="assets/Logo_Rwanda.png",
                    style={"width": "40px", "height": "40px", "margin-right": "10px", "vertical-align": "middle"}
                ),
                # html.H2("RY-Work", className="display-4", style={"display": "inline", "vertical-align": "middle"})
                html.H2("RwandaYouth-Work", className="display-4", style={"display": "inline", "vertical-align": "middle", "font-size": "24px"})
            ],
            style={"display": "flex", "align-items": "center"}
        ),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-home me-2 icon-home"), "Home"], href="/", id="home-link", active="exact"),
                dbc.NavLink([html.I(className="fas fa-chart-bar me-2 icon-visualization"), "Visualization"], href="/visualization", id="visualization-link", active="exact"),
                dbc.NavLink([html.I(className="fas fa-bullseye me-2 icon-prediction"), "Prediction"], href="/prediction", id="prediction-link", active="exact"),
                dbc.NavLink([html.I(className="fas fa-comment me-2 icon-feedback"), "Feedback"], href="/feedback", id="feedback-link", active="exact"),
            ],
            vertical=True, pills=True,
        ),
    ],
    className="sidebar",
)

# Main content area
content = html.Div(id="page-content", className="content-section")

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(content, width=9),
            ],
            id="main-row",
            style={"height": "100vh", "display": "flex"}
        ),]
)

# Callback to update page content based on URL pathname
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_content(pathname):
            
    if pathname == "/":
        return html.Div([
            # Hero section with background image and blue overlay
            html.Div([
                # Content container
                html.Div([
                    html.H1("Rwanda Youth Employment Initiative",
                        className="display-4 mb-3",
                        style={
                            'color': 'white',
                            'font-weight': 'bold',
                            'text-align': 'center',
                            # 'padding-top': '40px',
                            'text-shadow': '2px 2px 4px rgba(0,0,0,0.3)'
                        }),
                    html.H4("Empowering Through Data Analysis",
                        className="mb-4",
                        style={
                            'text-align': 'center',
                            'color': 'rgba(255,255,255,0.9)',
                            'text-shadow': '1px 1px 2px rgba(0,0,0,0.3)'
                        }),
                    html.Hr(style={
                        'width': '50%', 
                        'margin': '20px auto',
                        'border-color': 'rgba(255,255,255,0.3)'
                    }),
                        
                    # Overview card
                    html.Div([
                        html.Div([
                            html.H3("Project Overview", 
                                className="mb-4",
                                style={'color': '#2C3E50'}),
                            html.P([
                                "This project aims to mitigate unemployment in Rwanda through insightful data analysis, predictions, and the use of Large Language Models (LLM). ",
                                html.Br(), html.Br(),
                                "Our key objectives include:"
                            ], style={'font-size': '1.1rem'}),
                            html.Ul([
                                html.Li("Analyzing employment trends across Rwanda", className="mb-2"),
                                html.Li("Predicting future employment patterns", className="mb-2"),
                                html.Li("Providing data-driven insights", className="mb-2"),
                                html.Li("Supporting policy decisions with AI-powered analysis", className="mb-2")
                            ], style={'font-size': '1.1rem'}),
                                
                            # Quick access buttons
                            html.Div([
                                dbc.Button([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "View Analytics"
                                ], 
                                color="primary", 
                                href="/visualization",
                                className="me-3 mt-4"),
                                    
                                dbc.Button([
                                    html.I(className="fas fa-brain me-2"),
                                    "See Predictions"
                                ],
                                color="success",
                                href="/prediction",
                                className="mt-4")
                            ], style={'text-align': 'center'})
                        ], 
                        className="p-5",
                        style={
                            'background-color': 'rgba(255, 255, 255, 0.95)',
                            'border-radius': '10px',
                            'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                            'margin': '20px auto',
                            'max-width': '800px',
                            'height': '100%'  # Set the height to 100% to match the sidebar
                        })
                    ], style={'padding': '20px', 'height': '100%'})
                ])
            ], style={
                'background-image': 'linear-gradient(rgba(41, 128, 185, 0.85), rgba(44, 62, 80, 0.85)), url("/assets/rwanda-bg.jpg")',
                'background-size': 'cover',
                'background-position': 'center',
                'background-repeat': 'no-repeat',
                'height': '100%',
                'padding': '20px'
            })
        ])
    elif pathname == "/visualization":
        return  html.Div([
        html.H2("Unemployment Data Insights", className="content-title"),
        dcc.Dropdown(
            id="chart-selector",
            options=[
                {'label': 'Employment Status by Age Group', 'value': 'age_group'},
                {'label': 'Employment Status by Province', 'value': 'province'},
                {'label': 'Distribution of Usual Working Hours', 'value': 'working_hours'},
                {'label': 'Agricultural Work by Age Group', 'value': 'agricultural_work'},
                {'label': 'Distribution of Age', 'value': 'age_distribution'},
                {'label': 'Youth Unemployment by Age Group', 'value': 'youth_unemployment'},
                {'label': 'Youth Unemployment status', 'value': 'unemployment_status'},
                {'label': 'Unemployment Trend Over Time', 'value': 'unemployment_trend'},
                {'label': 'Unemployment by Educational Attainment', 'value': 'unemployment_education'},
                {'label': 'Unemployment by Gender', 'value': 'unemployment_gender'},
                {'label': 'Unemployment by District', 'value': 'unemployment_district'},
                {'label': 'Unemployment Forecast', 'value': 'unemployment_forecast'},
                {'label': 'Unemployment by HDI', 'value': 'hdi_vs_unemployment'}
            ],
            value="age_group",
            className="dropdown"
        ),
        html.Div([
            dcc.Dropdown(
                id="age-filter",
                options=[{'label': age_group, 'value': age_group} for age_group in data['age5'].unique()],
                placeholder="Filter by Age Group",
                className="filter-dropdown"
            ),
            dcc.Dropdown(
                id="province-filter",
                options=[{'label': province, 'value': province} for province in data['province'].unique()],
                placeholder="Filter by Province",
                className="filter-dropdown"
            ),

            dcc.Dropdown(
                id="education-filter",
                options=[{'label': edu_level, 'value':edu_level} for edu_level in data['attained'].unique()],
                placeholder="Filter by Education level",
                className="filter-dropdown"
            ),


        ], className="filter-section"),
        dcc.Graph(id="main-chart", className="main-chart"),
        
    ], id="dashboard-content", className="content")
        
    # elif pathname == "/prediction":
    #     return html.Div([html.H3("Predictions"), html.P("Predictions based on unemployment data and trends."),
                         
            
    #     ])
    # return html.Div(html.H3("Please select an option from the sidebar."))
    elif pathname == "/prediction":
        try:
            filtered_data = data.copy()
            # Step 1: Calculate unemployment trend by working year
            unemployment_trend = filtered_data.groupby('working_year').agg({
                'target_employed16': lambda x: 1 - x.mean()
            }).reset_index()
            unemployment_trend = unemployment_trend.rename(columns={'target_employed16': 'Unemployment_Rate'})
            unemployment_trend.set_index('working_year', inplace=True)

            # Step 2: Fit ARIMA model on the unemployment rate
            model = ARIMA(unemployment_trend['Unemployment_Rate'], order=(1, 1, 1))
            results = model.fit()

            # Step 3: Forecast for the next 25 years
            forecast_years = 25
            forecast = results.forecast(steps=forecast_years)
            
            # Step 4: Prepare forecast DataFrame with proper years
            forecast_start_year = unemployment_trend.index[-1] + 1
            forecast_df = pd.DataFrame({'Forecasted_Unemployment_Rate': forecast})
            forecast_df.index = pd.RangeIndex(start=forecast_start_year, 
                                            stop=forecast_start_year + forecast_years)

            # Step 5: Plot historical and forecasted data
            fig = px.line(unemployment_trend, x=unemployment_trend.index, y='Unemployment_Rate',
                        title="Historical and Forecasted Unemployment Rate")
            fig.add_scatter(x=forecast_df.index, y=forecast_df['Forecasted_Unemployment_Rate'],
                            mode='lines', name='Forecast', line=dict(dash='dash', color='red'))
            
            # Step 6: Customize the layout for better readability
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Unemployment Rate',
                xaxis=dict(tickmode='linear', dtick=1),  # Yearly ticks
                template="plotly_white"
            )

            # Return the layout with the chart
            return html.Div([
                html.H3("Unemployment Forecast"),
                dcc.Graph(figure=fig)  # Display the chart here
            ])

        except Exception as e:
            print(f"Error: {e}")
            return html.Div([
                html.H3("Error"),
                html.P("An error occurred while generating the unemployment forecast chart."),
                dcc.Graph(figure=px.scatter(x=[0], y=[0], title="Error occurred in generating the chart"))
            ])
    elif pathname == "/feedback":
            return html.Div([
        html.Div([
            html.H1("Feedback Dashboard", style={"textAlign": "center"}),
            dcc.Graph(id="feedback-chart"),
            dcc.Interval(id="update-interval", interval=60*1000, n_intervals=0),  # Update every minute
        ]),
        html.Div([
            dash_table.DataTable(
                id="feedback-table",
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Status", "id": "status"},
                    # {"name": "Email", "id": "email"},
                    {"name": "Message", "id": "message"},
                ],
                style_data={"whiteSpace": "normal", "height": "auto"},
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
                style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
            ),
            html.Button("Download CSV", id="download-btn", style={"marginTop": "10px"}),
            dcc.Download(id="download-dataframe-csv"),
        ]),
    ])
    return html.Div(html.H3("Please select an option from the sidebar."))

import os
import time
import boto3
from botocore.exceptions import ClientError
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region = os.getenv('AWS_REGION', 'us-west-2')
table_name = os.getenv('DYNAMODB_TABLE_NAME', 'Feedback')

dynamodb = boto3.client(
    'dynamodb',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)


def fetch_dynamodb_feedback():
    """
    Fetch feedback items from DynamoDB with retry logic.
    """
    retries = 3
    delay = 2  # seconds
    for attempt in range(retries):
        try:
            response = dynamodb.scan(TableName=table_name)
            items = response.get('Items', [])
            processed_items = [
                {
                    'email': item.get('email', {}).get('S', ''),
                    'id': item.get('id', {}).get('S', ''),
                    'message': item.get('message', {}).get('S', ''),
                    'status': int(item.get('status', {}).get('N', 0)),
                    'timestamp': item.get('timestamp', {}).get('S', '')
                }
                for item in items
            ]
            return processed_items
        except ClientError as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    print("All retries failed.")
    return []

def process_feedback_data(items):
    """
    Process feedback data into counts and a DataFrame.
    """
    df = pd.DataFrame(items)
    if not df.empty:
        df['status'] = df['status'].astype(int)
    positive_count = df[df['status'] == 1].shape[0] if not df.empty else 0
    negative_count = df[df['status'] == 0].shape[0] if not df.empty else 0
    return positive_count, negative_count, df

# Add a callback to handle feedback submission
@app.callback(
    Output("feedback-message", "children"),
    Input("submit-feedback", "n_clicks"),
    State("feedback-text", "value")
)
def handle_feedback_submission(n_clicks, feedback):
    if n_clicks:
        if feedback:
            # You can save the feedback to a database or log file here
            print(f"Feedback received: {feedback}")
            return "Thank you for your feedback!"
        return "Please enter your feedback before submitting."
    return ""
# Default message for unmatched paths
    return html.Div(html.H3("Please select an option from the sidebar."))

@app.callback(
    [Output("feedback-chart", "figure"),
     Output("feedback-table", "data")],
    [Input("update-interval", "n_intervals")]
)
def update_dashboard(n_intervals):
    items = fetch_dynamodb_feedback()
    positive_count, negative_count, df = process_feedback_data(items)

    # Update bar graph
    figure = {
        "data": [
            go.Bar(name="Positive Feedback", x=["Feedback"], y=[positive_count], marker_color="#0047AB"),
            go.Bar(name="Negative Feedback", x=["Feedback"], y=[negative_count], marker_color="#00AEEF"),
        ],
        "layout": go.Layout(
            title="Feedback Summary",
            barmode="stack",
            plot_bgcolor="#1E1E1E",
            paper_bgcolor="#1E1E1E",
            font={"color": "white"},
        )
    }

    # Update table data
    table_data = df.to_dict("records") if not df.empty else []

    return figure, table_data

@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-btn", "n_clicks")],
    prevent_initial_call=True
)
def download_csv(n_clicks):
    items = fetch_dynamodb_feedback()
    _, _, df = process_feedback_data(items)
    return dcc.send_data_frame(df.to_csv, "feedback.csv", index=False)

# Callback to update the main chart based on user selection
@app.callback(
    Output('main-chart', 'figure'),
    [Input('chart-selector', 'value'),
     Input('age-filter', 'value'),
     Input('province-filter', 'value'),
     Input('education-filter', 'value')]
     

)
def update_chart(selected_chart, age_filter, province_filter, education_filter):
    try:
        filtered_data = data.copy()
        hdi_data = data1.copy()
        unemployment_data = data2.copy()

        # Apply filters if selected
        if age_filter:
            filtered_data = filtered_data[filtered_data['age5'] == age_filter]
        if province_filter:
            filtered_data = filtered_data[filtered_data['province'] == province_filter]

        if education_filter:
            filtered_data = filtered_data[filtered_data['attained'] == education_filter]    

        if len(filtered_data) == 0:
            raise PreventUpdate

        # Generate the appropriate chart based on user selection
        if selected_chart == 'age_group':
            fig = px.histogram(filtered_data, x='age5', color='LFS_workforce', barmode='group',
                               title='Employment Status by Age Group',color_discrete_sequence=['#0047AB', '#00AEEF', '#FFED5F'])
            fig.update_layout(xaxis_title='Age Group', yaxis_title='Employment Rate',plot_bgcolor='white',
                              paper_bgcolor='white',template="plotly_white",
                              font=dict(color='#0047AB'))
        elif selected_chart == 'youth_unemployment':
                # Calculate average unemployment rate by age group
                average_unemployment = filtered_data.groupby('age5')['target_employed16'].mean()
                average_unemployment = 1 - average_unemployment  # Unemployment rate is 1 minus the employment rate

                # Create the figure
                fig = go.Figure()

                # Add bar plot for unemployment rate by age group
                fig.add_trace(go.Bar(
                    x=average_unemployment.index,
                    y=average_unemployment,
                    name='Unemployment Rate',
                    marker=dict(color='#0047AB', line=dict(color='#00AEEF', width=1)),
                ))

                # Add line plot on top of the bar chart
                fig.add_trace(go.Scatter(
                    x=average_unemployment.index,
                    y=average_unemployment,
                    mode='lines+markers',
                    line=dict(color='#FFED5F', width=2),
                    marker=dict(symbol='circle', size=8,color='#FFED5F' ),
                    name='Average Unemployment Rate'
                ))

                # Annotate bars with unemployment rate values
                for age_group, rate in zip(average_unemployment.index, average_unemployment):
                    fig.add_annotation(
                        x=age_group,
                        y=rate,
                        text=f'{rate:.2%}',
                        showarrow=False,
                        yshift=5,
                        font=dict(size=10, color="#0047AB")
                    )

                # Set titles and labels
                fig.update_layout(
                    title='Youth Unemployment Rate by Age Group',
                    xaxis_title='Age Group',
                    yaxis_title='Average Unemployment Rate',
                    yaxis=dict(tickformat='.0%', range=[0, max(average_unemployment) * 1.1]),  # Set y-axis format to percentage
                    xaxis=dict(tickmode='array', tickvals=average_unemployment.index),
                    template='plotly_white',
                    legend=dict(x=0.9, y=1.1)
                )

                # Customize x-axis for readability
                fig.update_xaxes(tickangle=45)

                # Add gridlines
                fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray', zeroline=False)



        elif selected_chart == 'unemployment_education':
            unemployment_by_education = 1 - filtered_data.groupby('attained')['target_employed16'].mean()
            fig = px.bar(x=unemployment_by_education.index, y=unemployment_by_education.values,
                         title='Unemployment Rates by Educational Attainment',
                         color=unemployment_by_education.index,
                         color_discrete_sequence=['#0047AB', '#00AEEF', '#FFED5F'])
            fig.update_layout(xaxis_title='Education Level', yaxis_title='Unemployment Rate',plot_bgcolor='white',
                               paper_bgcolor='white', template="plotly_white",
                               font=dict(color='#0047AB'))

        elif selected_chart == 'unemployment_gender':
            unemployment_by_gender = 1 - filtered_data.groupby('A01')['target_employed16'].mean()
            fig = px.bar(x=unemployment_by_gender.index, y=unemployment_by_gender.values,
                         title='Youth Unemployment Rate by Gender',color=unemployment_by_gender.index,
                            color_discrete_sequence=['#0047AB', '#00AEEF'])
            fig.update_layout(xaxis_title='Gender', yaxis_title='Unemployment Rate',legend_title_text='Gender',plot_bgcolor='white',
                               paper_bgcolor='white',template="plotly_white",
                                 font=dict(color='#0047AB'))

        elif selected_chart == 'unemployment_district':
            non_zero_unemployment = filtered_data[filtered_data['target_employed16'] == 0].groupby(['province', 'code_dis']).size().reset_index(name='Unemployed_Count')
            non_zero_unemployment = non_zero_unemployment[non_zero_unemployment['Unemployed_Count'] > 0]
            fig = px.bar(non_zero_unemployment, y='code_dis', x='Unemployed_Count', color='province',
                         title='Unemployment Counts by District within Provinces', orientation='h',color_discrete_sequence=['#0047AB', '#00AEEF', '#FFED5F','#E6194B', '#3CB44B'])
            fig.update_layout(yaxis_title='District', xaxis_title='Number of Unemployed',plot_bgcolor='white',
                              paper_bgcolor='white',template="plotly_white",
                              font=dict(color='#0047AB'))    

        elif selected_chart == 'province':
            # Calculate current unemployment rate by province
            unemployment_by_province = filtered_data.groupby('province').agg({
                'target_employed16': lambda x: 1 - x.mean()
            }).reset_index()
            unemployment_by_province = unemployment_by_province.rename(columns={'target_employed16': 'Employment_Rate'})

            # Create bar chart using Plotly
            fig = px.bar(
                unemployment_by_province,
                x='province',
                y='Employment_Rate',
                color='Employment_Rate',
                color_discrete_sequence=['#0047AB', '#00AEEF', '#FFED5F', '#3CB44B', '#E6194B'],
                labels={'province': 'Province', 'Employment_Rate': 'Employment Rate'},
                title='Employment Rate by Province'
            )
            fig.update_layout(
                xaxis_title="Province",
                yaxis_title="Employment Rate",
                xaxis_tickangle=45,
                legend_title_text='Province',
                plot_bgcolor='white',
                paper_bgcolor='white',
                 font=dict(color='#0047AB'),
                 template="plotly_white"
            )  

        elif selected_chart == 'unemployment_status':
            # Define counts and percentages for youth unemployment status within this block
            unemployment_counts = filtered_data['target_employed16'].value_counts()
            unemployment_counts.index = unemployment_counts.index.map(label_mapping).fillna('Unknown')
            
            total = unemployment_counts.sum()
            unemployment_percentages = unemployment_counts / total * 100

            # Create the pie chart for youth unemployment status
            fig = go.Figure(go.Pie(
                labels=unemployment_counts.index,
                values=unemployment_percentages,
                hole=0.3,
                marker=dict(colors=['#0047AB', '#00AEEF']),
                hoverinfo="label+percent",
                textinfo="percent",
                textfont_size=16
            ))

            # Update layout for the pie chart
            fig.update_layout(
                title='Youth Unemployment Status',
                legend=dict(title='Status', x=1, y=0.5),
                margin=dict(l=50, r=50, t=50, b=50),
                template="plotly_white",
                font=dict(color='#0047AB')
            )     

        elif selected_chart == 'working_hours':
            fig = px.histogram(filtered_data, x='usualhrs', nbins=20, title='Distribution of Usual Working Hours',color_discrete_sequence=['#0047AB'])
            fig.update_layout(xaxis_title='Usual Working Hours', yaxis_title='Frequency',plot_bgcolor='white',
                              paper_bgcolor='white',template="plotly_white",
                              font=dict(color='#0047AB'))

        elif selected_chart == 'agricultural_work':
            fig = px.histogram(filtered_data, x='age10', color='work_agr', barmode='group',
                            title='Agricultural Work by Age Group',color_discrete_sequence=['#0047AB', '#3CB44B','#00AEEF', '#FFED5F','#E6194B'])
            fig.update_layout(xaxis_title='Age Group', yaxis_title='Count', legend_title='Agricultural Work',plot_bgcolor='white',
                             paper_bgcolor='white',template="plotly_white",
                              font=dict(color='#0047AB'))

        elif selected_chart == 'age_distribution':
            fig = px.histogram(filtered_data, x='age5', title='Distribution of Age',color_discrete_sequence=['#0047AB'])
            fig.update_layout(xaxis_title='Age Group', yaxis_title='Count',plot_bgcolor='white',
                              paper_bgcolor='white',template="plotly_white",
                            font=dict(color='#0047AB'))   

        elif selected_chart == 'unemployment_trend':
            unemployment_trend = filtered_data.groupby('working_year').agg({
                'target_employed16': lambda x: 1 - x.mean()
            }).reset_index()
            unemployment_trend = unemployment_trend.rename(columns={'target_employed16': 'Unemployment_Rate'})
            fig = px.line(unemployment_trend, x='working_year', y='Unemployment_Rate',
                          title='Trend of Youth Unemployment Over Time')
            fig.update_layout(xaxis_title='Year', yaxis_title='Unemployment Rate',template="plotly_white")    

        elif selected_chart == 'unemployment_forecast':
            

            # Step 1: Calculate unemployment trend by working year
            unemployment_trend = filtered_data.groupby('working_year').agg({
                'target_employed16': lambda x: 1 - x.mean()
            }).reset_index()
            unemployment_trend = unemployment_trend.rename(columns={'target_employed16': 'Unemployment_Rate'})
            unemployment_trend.set_index('working_year', inplace=True)

            # Step 2: Fit ARIMA model on the unemployment rate
            model = ARIMA(unemployment_trend['Unemployment_Rate'], order=(1, 1, 1))
            results = model.fit()

            # Step 3: Forecast for the next 25 years
            forecast_years = 25
            forecast = results.forecast(steps=forecast_years)
            
            # Step 4: Prepare forecast DataFrame with proper years
            forecast_start_year = unemployment_trend.index[-1] + 1
            forecast_df = pd.DataFrame({'Forecasted_Unemployment_Rate': forecast})
            forecast_df.index = pd.RangeIndex(start=forecast_start_year, 
                                              stop=forecast_start_year + forecast_years)

            # Step 5: Plot historical and forecasted data
            fig = px.line(unemployment_trend, x=unemployment_trend.index, y='Unemployment_Rate',
                          title="Historical and Forecasted Unemployment Rate")
            fig.add_scatter(x=forecast_df.index, y=forecast_df['Forecasted_Unemployment_Rate'],
                            mode='lines', name='Forecast', line=dict(dash='dash', color='red'))
            
            # Step 6: Customize the layout for better readability
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Unemployment Rate',
                xaxis=dict(tickmode='linear', dtick=1),  # Yearly ticks
                template="plotly_white"
            )
        
        # Generate the appropriate chart based on user selection
        elif selected_chart == 'hdi_vs_unemployment':

            # Ensure the data for HDI and Unemployment is aligned properly
            x = hdi_data['Value']
            y = unemployment_data['Value']
    
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            trendline = slope * x + intercept

            # Create scatter plot with trendline using Plotly
            fig = go.Figure()

            # Add scatter plot for HDI vs Unemployment
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name='Data Points',
                marker=dict(color='cyan', size=8),
                hovertemplate='<b>HDI:</b> %{x}<br><b>Unemployment:</b> %{y}<extra></extra>'
            ))

            # Add trendline
            fig.add_trace(go.Scatter(
                x=x,
                y=trendline,
                mode='lines',
                name='Trendline',
                line=dict(color='red', width=2),
                hoverinfo='skip'
            ))

            # Update layout
            fig.update_layout(
                title='Relationship Between HDI and Unemployment',
                xaxis_title='Human Development Index (HDI)',
                yaxis_title='Unemployment (%)',
                template='plotly_white',
                legend=dict(x=0.8, y=1.1),
            )
                # Configure gridlines for better readability
            # fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
            # fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')


        else:
            # Placeholder chart for unhandled chart types
            fig = px.scatter(x=[0], y=[0])

        return fig

    except Exception as e:
        print(f"Error: {e}")
        return px.scatter(x=[0], y=[0], title="Error occurred in generating the chart")



@app.callback(
    Output("chat-output", "children"),  # Update chat message display
    Input("send-btn", "n_clicks"),  # Send button triggers the response
    State("chat-input", "value"),  # User input from chat
    State("chart-selector", "value"),  # Selected chart value
    State("chat-output", "children"),  # Chat history display
    prevent_initial_call=True
)
def update_chat(n_clicks, user_message, selected_chart, chat_history):
    # Ensure there is a message to process
    if not user_message:
        return chat_history or []

    # Format user's message and bot's response with the selected chart
    user_display = html.Div(f"You: {user_message}", className="user-message")
    prompt = f"Based on the '{selected_chart}' chart from the Labour Force Survey 2019, answer the following question: {user_message}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing labor force survey data."},
                {"role": "user", "content": prompt}
            ]
        )
        bot_reply = response.choices[0].message['content']
        bot_display = html.Div(f"Bot: {bot_reply}", className="bot-message")
        
        # Update chat history with the new messages
        chat_history = (chat_history or []) + [user_display, bot_display]

    except Exception as e:
        print(str(e))
        error_display = html.Div(f"An error occurred: {str(e)}", className="error-message")
        chat_history = (chat_history or []) + [user_display, error_display]

    # Return the updated chat history
    return chat_history

if __name__ == "__main__":
    app.run_server(debug=False)



######################################################################################################################

