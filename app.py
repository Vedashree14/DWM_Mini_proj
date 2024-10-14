import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load Data
def load_data():
    url = 'marketing_campaign.csv'
    df = pd.read_csv(url, delimiter='\t')
    df.columns = df.columns.str.strip()
    return df

# K-Means Clustering
def apply_kmeans(data, n_clusters=3):
    features = data[['Income', 'MntWines', 'MntFruits']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(scaled_features)
    return kmeans, data

# Random Forest Model
def train_random_forest(data):
    features = data[['Income', 'MntWines', 'MntFruits']]
    labels = data['Marital_Status']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    rf_model = RandomForestClassifier(random_state=0)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    return rf_model, predictions, y_test

df = load_data()
df.dropna(subset=['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], inplace=True)

# Initialize App with Tailwind CDN
app = Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"])

# App Layout with Professional Tailwind Styling
app.layout = html.Div([
    html.H1("Customer Insights Dashboard", className='text-center mt-9 mb-4 font-extrabold text-3xl text-gray-800'),
    html.Div([

        # Left Sidebar with Dropdowns using Tailwind classes
        html.Div([
            html.Label("Select Education Level:", className='mt-2 text-lg font-bold text-gray-700'),
            dcc.Dropdown(
                id='education-dropdown',
                options=[{'label': i, 'value': i} for i in df['Education'].unique()],
                multi=True,
                value=df['Education'].unique().tolist(),
                className='dropdown mb-6 text-lg border border-gray-300 rounded-md p-2 shadow-sm'
            ),
            html.Label("Select Marital Status:", className='mt-2 text-lg font-bold text-gray-700'),
            dcc.Dropdown(
                id='marital-status-dropdown',
                options=[{'label': i, 'value': i} for i in df['Marital_Status'].unique()],
                multi=True,
                value=df['Marital_Status'].unique().tolist(),
                className='dropdown mb-6 text-lg border border-gray-300 rounded-md p-2 shadow-sm'
            ),
            html.Div(id='summary-info', className='mt-6 p-4 bg-white rounded-lg shadow-lg border border-gray-200')
        ], className='w-1/4 p-6 bg-gray-50 h-screen border-r border-gray-200'),  # Sidebar width is 25%
        
        # Right Content Area for Graphs
        html.Div([
            dcc.Graph(id='income-distribution', className='mb-6 bg-white p-4 rounded-lg shadow-md'),
            dcc.Graph(id='customer-count', className='mb-6 bg-white p-4 rounded-lg shadow-md'),
            html.Div([
                dcc.Graph(id='avg-income-education', className='w-1/2 p-4 bg-white rounded-lg shadow-md'),
                dcc.Graph(id='total-spending', className='w-1/2 p-4 bg-white rounded-lg shadow-md'),
            ], className='flex'),
            dcc.Graph(id='customer-segmentation', className='mb-6 bg-white p-4 rounded-lg shadow-md'),
            dcc.Graph(id='random-forest-output', className='mb-6 bg-white p-4 rounded-lg shadow-md'),
            dcc.Graph(id='feature-importance-plot', className='mb-6 bg-white p-4 rounded-lg shadow-md'),
        ], className='w-3/4 p-6 bg-gray-100')
    ], className='flex')
], className='min-h-screen bg-gray-100')

# Update Graphs and Summary
@app.callback(
    [Output('income-distribution', 'figure'),
     Output('customer-count', 'figure'),
     Output('avg-income-education', 'figure'),
     Output('total-spending', 'figure'),
     Output('customer-segmentation', 'figure'),
     Output('random-forest-output', 'figure'),
     Output('feature-importance-plot', 'figure'),
     Output('summary-info', 'children')],
    [Input('education-dropdown', 'value'),
     Input('marital-status-dropdown', 'value')]
)
def update_graphs(education, marital_status):
    filtered_data = df[df['Education'].isin(education) & df['Marital_Status'].isin(marital_status)]
    
    # Dynamic Summary Information
    total_customers = len(filtered_data)
    avg_income = filtered_data['Income'].mean()

    summary_info = html.Div([
        html.H4(f'Total Customers: {total_customers}', className='text-xl font-bold text-gray-800'),
        html.H5(f'Average Income: ${avg_income:,.2f}', className='text-lg font-bold text-gray-600')
    ])

    # Income Distribution
    income_fig = px.histogram(filtered_data, x='Income', nbins=30, title=f'Income Distribution (Education: {education}, Marital Status: {marital_status})',
                              marginal='box', color_discrete_sequence=['#1f77b4'])

    # Customer Count
    customer_count_fig = px.histogram(filtered_data, x='Marital_Status', title='Count of Customers by Marital Status',
                                      color='Marital_Status', color_discrete_sequence=px.colors.qualitative.Plotly)

    # Avg Income by Education Level
    avg_income_fig = px.bar(filtered_data.groupby('Education')['Income'].mean().reset_index(),
                            x='Education', y='Income', title=f'Average Income by Education Level (Filtered)',
                            color='Income', color_continuous_scale=px.colors.sequential.Plasma)

    # Total Spending
    spending_categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    total_spending = filtered_data[spending_categories].sum().reset_index()
    total_spending.columns = ['Category', 'Total Spending']
    total_spending_fig = px.pie(total_spending, values='Total Spending', names='Category', title='Total Spending by Category')

    # K-Means Clustering
    kmeans_model, clustered_data = apply_kmeans(filtered_data)
    clustering_fig = px.scatter(clustered_data, x='Income', y='MntWines', color='Cluster',
                                title='K-Means Clustering', color_continuous_scale=px.colors.qualitative.Plotly)

    # Random Forest Confusion Matrix
    rf_model, rf_predictions, y_test = train_random_forest(filtered_data)
    cm = confusion_matrix(y_test, rf_predictions)
    cm_fig = px.imshow(cm, text_auto=True, title='Confusion Matrix', labels={'x': 'Predicted', 'y': 'Actual'})

    # Feature Importance
    feature_importances = rf_model.feature_importances_
    feature_names = ['Income', 'MntWines', 'MntFruits']
    feature_importance_fig = px.bar(x=feature_names, y=feature_importances, title='Feature Importance from Random Forest',
                                    color=feature_importances, color_continuous_scale=px.colors.sequential.Viridis)

    return income_fig, customer_count_fig, avg_income_fig, total_spending_fig, clustering_fig, cm_fig, feature_importance_fig, summary_info

if __name__ == '__main__':
    app.run_server(debug=True)
