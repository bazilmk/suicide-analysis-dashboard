# # Importing Libraries

# ''' Visualization Libaries '''
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import datetime

''' Data / Other Libraries '''
import numpy as np
import pandas as pd
import os
import math
import json

''' Classes Defined below '''

plot_h, plot_w, plot_m = 432, 920, 0 

class PlotlyVisualizations:
    ''' Class containing functions to create plots '''

    def world_map(self, df, x_col, y_col, label_col, color, chart_title):

        map_json = {

            'data': [go.Choropleth(
                locations=df[x_col],
                z=df[y_col],
                text=df[label_col],
                colorscale=color,
                autocolorscale=False,
                reversescale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title="Suicides per 100k people"
            )],

            'layout': go.Layout(
                # title_text=chart_title,
                # margin={'b': 40, 't': 50, 'r': 10},
                width=plot_w,
                height=plot_h,
                margin={'b': 0, 't': 0, 'r': 0, 'l': 0},
                clickmode='event+select',
                geo=dict(
                    showframe=False,
                    showcoastlines=False,
                    landcolor = 'rgb(230, 230, 230)',
                    showland = True,
                    # showcountries = True,
                )
            )
        }

        return map_json


    def boxplot(self, df, category=0, selected_countries={}):
        suicide_rates = df['total_suicide_rate'].to_numpy()
        countries, countries_u = df['country'].to_numpy(), set(df['country'].dropna().unique())

        if category == 0:
            groups = df['Region'].to_numpy()
        else:
            groups = df[category].to_numpy()

        fig = go.Figure()
        if category == 0:
            boxes, selected_countries_data = dict(), []
            for c, g, sr in zip(countries, groups, suicide_rates):
                if g in boxes:
                    boxes[g].append(sr)
                else:
                    boxes[g] = [sr]
                if c in selected_countries:
                    selected_countries_data.append(sr)

            add_selected_countries = False
            if len(selected_countries_data) > 1:
                add_selected_countries = True
                y_selected_countries = np.percentile(selected_countries_data, (5, 25, 25, 50, 75, 75, 95)) 
                x_selected_countries = ['Selected Countries']*7
        else:
            boxes, boxes_group = dict(), dict() 
            for c, g, sr in zip(countries, groups, suicide_rates):
                if g in boxes:
                    boxes[g].append(sr)
                else:
                    boxes[g] = [sr]
                if c in selected_countries:
                    if g in boxes_group:
                        boxes_group[g].append(sr)
                    else:
                        boxes_group[g] = [sr]
                

        x, y = [], []
        for k, v in boxes.items():
            percentiles = np.percentile(v, (5, 25, 25, 50, 75, 75, 95)) 
            y.extend(list(map(str, percentiles)))
            x.extend([k]*7)
        if category != 0 and len(selected_countries) > 0:
            x_group, y_group = [], []
            for k, v in boxes_group.items():
                percentiles = np.percentile(v, (5, 25, 25, 50, 75, 75, 95)) 
                y_group.extend(list(map(str, percentiles)))
                x_group.extend([k]*7)
        
        # Add traces
        if category == 0:
            if add_selected_countries:
                x = np.concatenate((x, x_selected_countries))
                y = np.concatenate((y, y_selected_countries))
            fig.add_trace(go.Box(
                x = x,
                y = y,
                boxpoints = False,
                marker_color='rgb(178,10,28)'
            ))
        else:
            if len(selected_countries) > 0:
                fig.add_trace(go.Box(
                    x = x,
                    y = y,
                    boxpoints = False,
                    name='All Countries',
                    marker_color='rgb(178,10,28)'
                ))
                fig.add_trace(go.Box(
                    x = x_group,
                    y = y_group,
                    boxpoints = False,
                    name='Selected Countries',
                    marker_color='rgb(245,160,105)'
                ))
            else:
                fig.add_trace(go.Box(
                    x = x,
                    y = y,
                    boxpoints = False,
                    marker_color='rgb(178,10,28)'
                ))

        fig.update_layout(
            yaxis_title='Suicides per 100k people',
            boxmode='group', # group together boxes of the different traces for each value of x
            width=plot_w,
            height=plot_h,
            margin={'b': 0, 't': 0, 'r': 0},
        )
        # fig.update_traces(orientation='v')
        
        return fig

    def scatter_plot(self, df, x_col, y_col, category_filter, label_col, chart_title, x_label):
        regions_trace_list = []
        for i in df[category_filter].unique():
            df_by_region = df[df[category_filter] == i]
            regions_trace_list.append(go.Scatter(x=df_by_region[x_col],
                                             y=df_by_region[y_col],
                                             text=df_by_region[label_col],
                                            #  mode='markers' if label_col == 'Region' else 'lines+markers',
                                             mode='markers',
                                             opacity=0.7,
                                             marker={
            'size': 10,
            'line': {'width': 1, 'color': 'white'}
        },
            name=i,
            ))

        scatter_json = {
            'data': regions_trace_list,
            'layout': go.Layout(
                # title_text=chart_title,
                xaxis={'title': x_label},
                yaxis={'title': 'Suicides per 100k people'},
                showlegend=True,
                hovermode='closest',
                transition={'duration': 500},
                width=plot_w,
                height=plot_h,
                margin={'t': 0, 'r': 0},
            )
        }

        return scatter_json

    def parallel_coordinates(self, df, color, labels, exclude=[], exclude_cat=[], select_cat=0, chart_title='Parallel Coordinates'):
        countries = df['country'].to_numpy()
        regions, regions_u = df['Region'].to_numpy(), list(df['Region'].dropna().unique())
        ages, ages_u = df['age'].to_numpy(), list(df['age'].dropna().unique())
        genders, genders_u = df['sex'].to_numpy(), list(df['sex'].dropna().unique())
        income_groups, income_groups_u = df['IncomeGroup'].to_numpy(), list(df['IncomeGroup'].dropna().unique())

        unemp = df['total_unemployment_rate'].to_numpy()
        life_expectancy = df['life_expectancy'].to_numpy()
        suicide_rates = df['total_suicide_rate'].to_numpy()

        age_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
        income_groups_oder = ['High income', 'Upper middle income', 'Lower middle income']
        ages_u = [age for age in age_order if age in ages_u]
        income_groups_u = [income_group for income_group in income_groups_oder if income_group in income_groups_u]
        regions_u.sort()
        genders_u.sort()

        dimensions = []
        try:
            dimensions.append({
                'range': [int(life_expectancy.min()), int(math.ceil(life_expectancy.max()))],
                'label': 'Life expectancy (year)',
                'values': life_expectancy
            })
        except:
            pass
        try:
            if len(genders_u) > 1:
                dimensions.append({
                    'range': [-1.5, len(genders_u)-1+1],
                    'tickvals': list(range(len(genders_u))),
                    'ticktext': genders_u,
                    'label': "Gender",
                    'values': ordered_encoding(genders, genders_u)
                })
        except:
            pass
        try:
            if len(regions_u) > 1:
                dimensions.append({
                    'range': [0, len(regions_u)-1],
                    'tickvals': list(range(len(regions_u))),
                    'ticktext': regions_u,
                    'label': "Region",
                    'values': ordered_encoding(regions, regions_u)
                })
        except:
            pass
        try:
            if len(income_groups_u) > 1:
                dimensions.append({
                    'range': [-1.5, len(income_groups_u)-1+1],
                    'tickvals': list(range(len(income_groups_u))),
                    'ticktext': income_groups_u,
                    'label': "Income group",
                    'values': ordered_encoding(income_groups, income_groups_u)
                })
        except:
            pass
        try:
            dimensions.append({
                'range': [0, int(math.ceil(unemp.max()))],
                'label': 'Unemployment rate (%)',
                'values': unemp
            })
        except:
            pass
        try:
            if len(ages_u) > 1:
                dimensions.append({
                    'range': [0, len(ages_u)-1],
                    'tickvals': list(range(len(ages_u))),
                    'ticktext': ages_u,
                    'label': "Age",
                    'values': ordered_encoding(ages, ages_u)
                })
        except:
            pass
        dimensions.append({
            'range': [suicide_rates.min(), suicide_rates.max()],
            'label': 'Suicides per 100k people',
            'values': suicide_rates
        })
        
        color_values = suicide_rates.copy()
        if exclude:
            color_values = np.where(np.isin(countries, exclude), color_values, -100)
        if exclude_cat:
            if select_cat == 0:
                if exclude_cat[0] in regions_u:
                    print(0, select_cat, exclude_cat)
                    color_values = np.where(np.isin(regions, exclude_cat), -100, color_values)
                else:
                    print(1, select_cat, exclude_cat)
                    color_values = np.where(np.isin(countries, exclude_cat), -100, color_values)
            elif select_cat == 'age':
                color_values = np.where(np.isin(ages, exclude_cat), -100, color_values)
            elif select_cat == 'sex':
                color_values = np.where(np.isin(genders, exclude_cat), -100, color_values)
            elif select_cat == 'IncomeGroup':
                color_values = np.where(np.isin(income_groups, exclude_cat), -100, color_values)
        
        cmax = int(math.ceil(suicide_rates.max()))
        cmin = cmax - cmax/0.990
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(
                    color = color_values,
                    colorscale=[
                        [0, 'rgb(230,230,230)'],
                        [0.01, 'rgb(230,230,230)'],
                        [0.01, 'rgb(255, 244, 238)'],
                        [0.2, 'rgb(245,195,157)'],
                        [0.4, 'rgb(245,160,105)'],
                        [1, 'rgb(178,10,28)']
                    ],
                    showscale = True,
                    cmin = cmin,
                    cmax = cmax
                ),
                dimensions = dimensions
            ),
            layout = dict(
                width=plot_w,
                height=plot_h,
                # margin={'b': 10, 't': 10, 'r': 10, 'l': 10}
            )
        )
        
        return fig


class DataPreperation:
    ''' Class containing functions which do the data preparation related tasks'''

    def __init__(self):
        self.path = os.getcwd()
        self.dataset_path = os.path.join(
            self.path, 'merged_suicide_rates_dataset.csv')
        self.suicide_rates_df = pd.read_csv(self.dataset_path)

    def get_df_complete(self):
        df_complete = self.suicide_rates_df
        df_complete['year'] = df_complete['year'].astype(
            str)
        df_complete['year'] = pd.to_datetime(
            df_complete['year'])
        df_complete['year'] = df_complete['year'].apply(
            lambda x: int(x.year))
        df_complete['life_expectancy'] = df_complete.apply(
            lambda row: row['male_life_expectancy_years'] if row['sex'] == 'male' else row['female_life_expectancy_years'],
            axis=1
        )
        return df_complete

    def get_unique_values(self, df):
        columns = ['year', 'country', 'country_code', 'Region', 'sex', 'age', 'IncomeGroup']
        return {col: df[col].dropna().unique() for col in columns}

    def agg_data_scatter_plot(self, df_input, select_y_axis, select_x_axis, select_category, country_list):
        df = df_input.copy()
        if country_list is None or country_list == []:
            if select_x_axis != 'year':
                if select_category != 0:
                    df = df.groupby(['Region', select_category]).agg(
                        {select_y_axis: sum, select_x_axis: 'mean', 'population': sum}).reset_index()
                else:
                    df = df.groupby(['Region']).agg(
                        {select_y_axis: sum, select_x_axis: 'mean', 'population': sum}).reset_index()
                df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000
            else:
                if select_category != 0:
                    df = df.groupby(['Region', select_x_axis, select_category]).agg(
                        {select_y_axis: sum, 'population': sum}).reset_index()
                else:
                    df = df.groupby(['Region', select_x_axis]).agg(
                        {select_y_axis: sum, 'population': sum}).reset_index()
                df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000
        else:
            df = df[df['country'].isin(country_list)]
            if select_x_axis != 'year':
                if select_category != 0:
                    df = df.groupby(['country', 'country_code', select_category]).agg(
                        {select_y_axis: sum, select_x_axis: 'mean', 'population': sum}).reset_index()
                else:
                    df = df.groupby(['country', 'country_code']).agg(
                        {select_y_axis: sum, select_x_axis: 'mean', 'population': sum}).reset_index()
                df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000
            else:
                if select_category != 0:
                    df = df.groupby(['country', 'country_code', select_x_axis, select_category]).agg(
                        {select_y_axis: sum, 'population': sum}).reset_index()
                else:
                    df = df.groupby(['country', 'country_code', select_x_axis]).agg(
                        {select_y_axis: sum, 'population': sum}).reset_index()
            df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000
        return df


def agg_data_world_map(df_input):
    df = df_input.copy()
    df = df.groupby(['country', 'country_code']).agg({'suicides_no': sum, 'population': sum}).reset_index() #'Region', 'IncomeGroup'
    df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000

    return df

def agg_data_parallel_coordinates(df_input):
    df = df_input.copy()
    df = df.dropna(
        subset=['age', 'sex', 'Region', 'country', 'country_code', 'suicides_no', 'population', 'life_expectancy', 'gdp_per_capita ($)', 'total_unemployment_rate'],
        how='any',
        axis=0
    )
    df = df.groupby(
        ['age', 'sex', 'Region', 'country', 'country_code', 'IncomeGroup']
    ).agg({
        'suicides_no': sum,
        'population': sum,
        'life_expectancy': 'first',
        'gdp_per_capita ($)': 'first',
        'total_unemployment_rate': 'first'
    }).reset_index()
    df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000

    return df

def agg_data_boxplot(df_input, select_x_axis='Region', country_code=None):
    df = df_input.copy()
    df['total_suicide_rate'] = df['suicides_no'] / df['population'] * 100000

    df = df[['country', select_x_axis, 'sex', 'age', 'IncomeGroup', 'total_suicide_rate']]
    
    return df


def ordered_encoding(array, order):
    y = list(order)
    return [y.index(x) for x in array]


# Create class objects
plotly_obj = PlotlyVisualizations()
data = DataPreperation()

# Call class functions
start_year = 1991
end_year = 2016
df_complete = data.get_df_complete()
unique_values = data.get_unique_values(df_complete)
label_dict = {'year':'Year', 'gdp_per_capita ($)': 'GDP per capita ($)', 'HDI for year':'Human Development Index', 'female_to_male_labor_rate':'Female to male labor rate', 'male_life_expectancy_years':'Male life expectancy', 'female_life_expectancy_years':'Female life expectancy', 'total_unemployment_rate':'Unemployment Rate (%)'}

''' Layout Code '''
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H1(
                                            children='Suicide Analysis',
                                            style={
                                                'fontSize': 32,
                                                'textAlign': 'center'
                                            }
                                        )
                                    ],
                                )
                            ],
                            className='row'
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id='select_x-axis',
                                            options=[
                                                {'label':'Year', 'value':'year'},
                                                {'label':'GDP per capita ($)', 'value':'gdp_per_capita ($)'},
                                                {'label':'Human Development Index', 'value':'HDI for year'},
                                                {'label':'Female to male labor rate', 'value':'female_to_male_labor_rate'},
                                                {'label':'Male life expectancy (y)', 'value':'male_life_expectancy_years'},
                                                {'label':'Female life expetancy (y)', 'value':'female_life_expectancy_years'},
                                                {'label':'Unemployment Rate (%)', 'value':'total_unemployment_rate'}
                                            ],
                                            value='year',
                                            multi=False,
                                            style={'fontSize': 12}
                                        )
                                    ],
                                    className='six columns',
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id='select_category',
                                            options=[
                                                {'label': 'Region', 'value': 0, 'disabled': False},
                                                {'label': 'Age', 'value': 'age', 'disabled': False},
                                                {'label': 'Gender', 'value': 'sex', 'disabled': False},
                                                {'label': 'Income Group', 'value': 'IncomeGroup', 'disabled': False}],
                                            value=0,
                                            multi=False,
                                            style={'fontSize':12}
                                        )
                                    ],
                                    className='six columns',
                                    id='dropdown-div'

                                )
                            ],
                            className='row'
                        ),
                    ],
                    className='three columns'
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    'Region',
                                ),
                                dcc.Dropdown(
                                    id='filter_region',
                                    options=[
                                        {'label': option, 'value': option} for option in unique_values['Region']
                                    ],
                                    value=unique_values['Region'],
                                    multi=True,
                                    placeholder="Select a region",
                                    style={'fontSize': 12}
                                )
                            ],
                            # style={'border':'2px black solid'},
                            className='five columns'
                        ),
                        html.Div(
                            [
                                html.P(
                                    'Income Group',
                                ),
                                dcc.Dropdown(
                                    id='filter_ig',
                                    options=[
                                        {'label': option, 'value': option} for option in unique_values['IncomeGroup']
                                    ],
                                    value=unique_values['IncomeGroup'],
                                    multi=True,
                                    placeholder="Select an income group",
                                    style={'fontSize': 12}
                                )
                            ],
                            # style={'border':'2px black solid'},
                            className='three columns'
                        ),
                        html.Div(
                            [
                                html.P(
                                    'Age Group',
                                ),
                                dcc.Dropdown(
                                    id='filter_age',
                                    options=[
                                        {'label': option, 'value': option} for option in unique_values['age']
                                    ],
                                    value=unique_values['age'],
                                    multi=True,
                                    placeholder="Select an age group",
                                    style={'fontSize': 12}
                                )
                            ],
                            className='three columns'
                        ),
                        html.Div(
                            [
                                html.P(
                                    'Gender',
                                ),
                                dcc.Checklist(
                                    id='filter_gender',
                                    options=[
                                        {'label': option.title(), 'value': option} for option in unique_values['sex']
                                    ],
                                    value=unique_values['sex'],
                                    style={'fontSize': 12}
                                )
                            ],
                            className='one column'
                        ),
                    ],
                    className='nine columns'
                ),
            ],
            className='row'
        ),        
        html.Div(
            [
                html.Div(dcc.RangeSlider(
                       id='range-slider',
                       min=start_year,
                       max=end_year,
                       value=[start_year, end_year],
                       marks={str(year): str(
                           year) for year in unique_values['year']},
                       ),
                       style={'width': '1500px',
                              'text-align': 'center', 'margin': 'auto'},
                       )
            ],
            style={'marginTop': '2px'},
            className='row'
        ),

        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id='world-map',
                        clickData=None,
                    ),
                    style={
                        'width': plot_w,
                        'margin-left': 960 - plot_w - int(plot_m//2),
                        'margin-right': plot_m,
                        'display': 'inline-block',
                        # 'border': '2px black solid'
                    }
                ),
                html.Div(                    
                    dcc.Graph(
                        id='parallel-coordinates',
                        figure=plotly_obj.parallel_coordinates(
                            df=agg_data_parallel_coordinates(df_complete),
                            color='total_suicide_rate',
                            labels={
                                'age': 'Age',
                                'sex': 'Gender',
                                'Region': 'Region',
                                'Population': 'Population',
                                'gdp_per_capita ($)': 'GDP per Capital ($)',
                                'total_suicide_rate': 'Suicides per 100k people'
                            }
                        )
                    ),
                    style={
                        'width': plot_w,
                        'display': 'inline-block',
                        # 'border': '2px black solid'
                    }
                )
            ],
            style={
                'display': 'inline-block'
            },
            className='row'
        ),
        html.Div(
            [
                html.Div(
                    dcc.Graph(
                        id='scatter-plot'
                    ),
                    style={
                        'width': plot_w,
                        'margin-left': 960 - plot_w - int(plot_m//2),
                        'margin-right': plot_m,
                        'display': 'inline-block',
                        # 'border': '2px black solid'
                    }
                ),
                html.Div(
                    dcc.Graph(
                        id='box-plot',
                        figure=plotly_obj.boxplot(
                            df=agg_data_boxplot(df_complete)
                        )
                    ),
                    style={
                        'width': plot_w,
                        'display': 'inline-block',
                        # 'border': '2px black solid'
                    }
                )
            ],
            style={
                'display': 'inline-block'
            },
            className='row'
        ),
        html.Div(id='intermediate-value', style={'display': 'none'})
    ],
)


@app.callback(
    Output('intermediate-value', 'children'),
    [
        dash.dependencies.Input('filter_region', 'value'),
        dash.dependencies.Input('filter_ig', 'value'),
        dash.dependencies.Input('filter_age', 'value'),
        dash.dependencies.Input('filter_gender', 'value'),
        dash.dependencies.Input('range-slider', 'value')
    ]
)
def clean_data(filter_region, filter_ig, filter_age, filter_gender, range_slider):
    df = df_complete.copy()
    df = df[ 
        (df['year'] >= range_slider[0]) & 
        (df['year'] <= range_slider[1]) &
        (df['Region'].isin(filter_region)) &
        (df['IncomeGroup'].isin(filter_ig)) &
        (df['age'].isin(filter_age)) &
        (df['sex'].isin(filter_gender))
    ]

    return df.to_json(orient='split')


@app.callback(
    Output('dropdown-div', 'children'),
    [
        Input('select_x-axis', 'value'),
    ])
def update_type(value):
    if value != 'year':
        return dcc.Dropdown(
            id='select_category',
            options=[
                {'label': 'Region', 'value': 0, 'disabled': False},
                {'label': 'Age', 'value': 'age', 'disabled': True},
                {'label': 'Gender', 'value': 'sex', 'disabled': True},
                {'label': 'Income Group', 'value': 'IncomeGroup', 'disabled': True}],
            value=0,
            multi=False,
            style={'fontSize': 12}
        )
    else:
        return dcc.Dropdown(
            id='select_category',
            options=[
                {'label': 'Region', 'value': 0, 'disabled': False},
                {'label': 'Age', 'value': 'age', 'disabled': False},
                {'label': 'Gender', 'value': 'sex', 'disabled': False},
                {'label': 'Income Group', 'value': 'IncomeGroup', 'disabled': False}],
            value=0,
            multi=False,
            style={'fontSize': 12}
        )

@app.callback(
    dash.dependencies.Output('world-map', 'figure'),
    [
        dash.dependencies.Input('intermediate-value', 'children'),
        dash.dependencies.Input('range-slider', 'value')
    ]
)
def update_world_map(json, range_slider):
    df = pd.read_json(json, orient='split')
    df = agg_data_world_map(df)
    chart_title = f"Suicide Rate per 100k people ({range_slider[0]} - {range_slider[1]})"
    
    return plotly_obj.world_map(
        df, 'country_code', 'total_suicide_rate', 'country', 
        [[0, 'rgb(255, 244, 238)'], [0.2, 'rgb(245,195,157)'], [0.4, 'rgb(245,160,105)'], [1, 'rgb(178,10,28)']],
        chart_title)

#
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('intermediate-value', 'children'),
        dash.dependencies.Input('select_x-axis', 'value'),
        dash.dependencies.Input('range-slider', 'value'),
        dash.dependencies.Input('select_category', 'value'),
        dash.dependencies.Input('world-map', 'clickData'),
        dash.dependencies.Input('world-map', 'selectedData'),
    ]
)
def update_scatter_plot(json, select_x_axis, range_slider, select_category, clickData, selectedData):
    country_list = []
    if selectedData != None:
        for i in range(len(selectedData['points'])):
            country_list.append(selectedData['points'][i]['text'])
        df = pd.read_json(json, orient='split')
        df = data.agg_data_scatter_plot(df, 'suicides_no', select_x_axis, select_category, country_list)
        chart_title = f"Suicide Rate per 100k people ({range_slider[0]} - {range_slider[1]})"
        if select_category == 0:
            select_category = 'country'
        if select_x_axis in label_dict:
            label = label_dict[select_x_axis]
        return plotly_obj.scatter_plot(df, select_x_axis, 'total_suicide_rate', select_category, 'country', chart_title, label)
    else:
        df = pd.read_json(json, orient='split')
        df = data.agg_data_scatter_plot(df, 'suicides_no', select_x_axis, select_category, None)
        chart_title = f"Suicide Rate per 100k people ({range_slider[0]} - {range_slider[1]})"
        if select_category == 0:
            select_category = 'Region'
        if select_x_axis in label_dict:
            label = label_dict[select_x_axis]
        return plotly_obj.scatter_plot(df, select_x_axis, 'total_suicide_rate', select_category, 'Region', chart_title, label)

@app.callback(
    dash.dependencies.Output('parallel-coordinates', 'figure'),
    [
        dash.dependencies.Input('intermediate-value', 'children'),
        dash.dependencies.Input('world-map', 'clickData'),
        dash.dependencies.Input('world-map', 'selectedData'),
        dash.dependencies.Input('scatter-plot', 'clickData'),
        dash.dependencies.Input('scatter-plot', 'selectedData'),
        dash.dependencies.Input('scatter-plot', 'restyleData'),
        dash.dependencies.Input('scatter-plot', 'figure'),
        dash.dependencies.Input('select_category', 'value')
    ]
)
def update_parallel_coordinates(json, clickData, selectedData, clickData_scat, selectedData_scat, extendData_scat, config_scat, select_category):  
    df = pd.read_json(json, orient='split')
    country_list = []
    if selectedData != None:
        for i in range(len(selectedData['points'])):
            country_list.append(selectedData['points'][i]['text'])

    scat_data, exclude_cat = config_scat['data'], []
    for cat in scat_data:
        if 'visible' in cat and cat['visible'] == 'legendonly':
            exclude_cat.append(cat['name'])

    return plotly_obj.parallel_coordinates(
        df=agg_data_parallel_coordinates(df),
        color='total_suicide_rate',
        labels={
            'age': 'Age',
            'sex': 'Gender',
            'Region': 'Region',
            'Population': 'Population',
            'gdp_per_capita ($)': 'GDP per Capital ($)',
            'total_suicide_rate': 'Suicides per 100k people',
        },
        exclude=country_list,
        exclude_cat=exclude_cat,
        select_cat=select_category
    )

@app.callback(
    dash.dependencies.Output('box-plot', 'figure'),
    [
        dash.dependencies.Input('intermediate-value', 'children'),
        dash.dependencies.Input('select_category', 'value'),
        dash.dependencies.Input('world-map', 'clickData'),
        dash.dependencies.Input('world-map', 'selectedData')
    ]
)
def update_boxplot(json, category, clickData, selectedData):
    df = pd.read_json(json, orient='split')
    country_list = []
    if selectedData != None:
        for i in range(len(selectedData['points'])):
            country_list.append(selectedData['points'][i]['text'])

    return plotly_obj.boxplot(
        df=agg_data_boxplot(df),
        category=category,
        selected_countries=set(country_list)
    )


if __name__ == '__main__':
    app.run_server(debug=False)