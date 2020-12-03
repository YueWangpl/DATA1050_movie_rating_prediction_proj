import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

import pandas as pd
import web_core
from preprocessing import *
from language_processing import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np


model_collect = {'model': 0, 'inference': 0}
ws_engine = web_core.WebSearch()

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
external_stylesheets = ['https://unpkg.com/wingcss']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


# Load Cleaned Dataframe:
df1 = pd.read_json('https://raw.githubusercontent.com/YueWangpl/DATA1050_movie_rating_prediction_proj/main/pop_movies.json')
df2 = df1[~(df1['year'] < 2015)]



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# PROJECT MAIN PAGE: NEED ADDING CONTRIBUTIONS, FUNCTION SPECIFICATIONS, DETAIL SAMPLES etc...
index_page = html.Div([

    html.H1("Movies", style={'text-align': 'center'}),
    html.Div([
    html.Div(children='''
        Graph showing relationship between score and vote numbers of different genres of movies.
    '''),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018},
                     {"label": "2019", "value": 2019},
                     {"label": "2020", "value": 2020}],
                 multi=False,
                 value=2015,
                 style={'width': "50%"}
                 ),
    html.Br(),
    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='movie_scatter', figure={})],style={'width': '75%', 'display': 'inline-block'}),
    html.Div( #smaller now moved up beside the first block
    [
        html.I("Search a movie to view its score trend, and try our prediction model!"),
        html.Br(),
        dcc.Link(html.Button('Search'), href='/search', style={'textAlign': 'center'}),
        html.Br()],
        style={'width': '20%', 'display': 'inline-block', 'margin':'auto','textAlign': 'center'}),

    html.Div(html.H2("Movie info per genre", style={'text-align': 'center'})),

    html.Div([
        dcc.Markdown('''
        This table shows 9502 trending movies from all genre.
        
        Use the filter box in each column to filter data.
        
        Genres include:
        *action, adventure, animation, biography, comedy, crime, 
        documentary, drama, family, fantasy, film-noir, history, horror, music, musical, 
        mystery, romance, sci-fi, short, sport, thriller, war, western*
        '''),  
        dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True}
            for i in df1.columns
        ],

        data=df1.to_dict('records'),  # the contents of the table
        filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        row_deletable=True,         # choose if user can delete a row (True) or not (False)
        page_action="native",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=8,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'textAlign': 'left','minWidth': 95, 'maxWidth': 95, 'width': 95
        },
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto',
        'lineHeight': '15px'
        }
    ),html.Br()]),
    html.Hr(),
    html.Div([
        
        html.Br(),
        dcc.Tabs([
            dcc.Tab(label='Action', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'action'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )
            ]),
            dcc.Tab(label='Adventure', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'adventure'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Animation', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'animation'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Biography', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'biography'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Comedy', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'comedy'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Crime', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'crime'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Drama', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'drama'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Family', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'family'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),

            dcc.Tab(label='Fantasy', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'fantasy'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Horror', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'horror'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Music', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'music'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )]),
            dcc.Tab(label='Romance', children=[
                dcc.Graph(
                    figure=px.scatter(df1[df1['genre'] == 'romance'], x="length", y="score", marginal_y="violin",
           marginal_x="box", trendline="ols", template="simple_white")
                )
            ])
        ])
    ])
])
# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='movie_scatter', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):

    container = "Year of {}".format(option_slctd)

    dff = df2.copy()
    dff = dff[dff["year"] == option_slctd]

    # Plotly Express
    fig = px.scatter(dff, x="vote_numbers", y="score",
                 size="gross", color="genre", hover_name="title",
                 log_x=True, size_max=60)

    return container, fig


# Search Page
page_1_layout = html.Div([
    html.H1('Search For a Movie', style={'text-align': 'center'}),
    html.Div(dcc.Input(id='input-on-submit', type='text', placeholder="Type a movie title here...")),
    html.Button('Search!', id='submit-val', n_clicks=0),
    dcc.Loading(id="loading", type="default",children="try", color='#708090'),
#     html.Div(id='container-button-basic',
#              children='Enter a value and press submit'),
    html.Div(id='search-content'),
    html.Br(),
    dcc.Link('HOME', href='/')
],style={'text-align': 'center'})


@app.callback(
    dash.dependencies.Output('loading', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    if not value:
        return ""
    else:
        search_result = ws_engine.search(value)
        if search_result:
            res = dict(zip(['movie_title', 'year', 'grade', 'length', 'genre', 'score', 'metascore', 'votes',
                            'gross', 'director', 'actor', 'related_movies', 'storyline'],
                           list(search_result.values())[0][0:13]))
            reviews = list(search_result.values())[0][13]  # Extracting the reviews
            scores = [int(x[0]) for x in reviews]  # Extracting the scores as training label
            text = [x[1] for x in reviews]
            pr = ProcessR(text)
            dtrain = xgb.DMatrix(np.array(pr.doc_vs), np.array(scores))
            param = {'max_depth': 60, 'eta': 0.03, 'objective': 'reg:squarederror', 'booster': 'gbtree'}
            evallist = [(dtrain, 'train')]
            model = xgb.train(param, dtrain, 200, evallist)
            model_collect['model'] = model
            model_collect['inference'] = pr

            # Trend plot
            date = [x[2] for x in reviews]

            ddd = {'date':date,'score':scores}
            df = pd.DataFrame(ddd)
            df['date'] = pd.to_datetime(df['date'])
            df = df.astype({'score': 'float'})
            df = df.sort_values(by=['date'])
            df['cum-avg'] = df['score'].expanding().mean().to_list()
            # Plot 
            trend_fig = px.line(df, x='date', y=['cum-avg'])
            # add another
            trend_fig.add_scatter(x=df['date'], y=df['score'],mode='markers', name='Individual Score')
            trend_fig.update_layout( height=800, title_text='Score Trend Over Time')
        else:
            return ""
    new_page_layout = html.Div([
        html.H4('Movie Title: {}'.format(res['movie_title'])),
        html.H4('Movie Released Year: {}'.format(res['year'])),
        dcc.Graph(figure=trend_fig),
        html.Div(dcc.Input(id='input-review', type='text')),
        html.Button('Predict!', id='submit-review', n_clicks=0),
        html.Div(id='review-button-basic',
                 children='Enter a review and predict the rating score.'),
        html.Br()])
    return new_page_layout


@app.callback(
    dash.dependencies.Output('review-button-basic', 'children'),
    [dash.dependencies.Input('submit-review', 'n_clicks')],
    [dash.dependencies.State('input-review', 'value')])
def update_review(n_clicks, value):
    if not model_collect['inference'] or not model_collect['model']:
        return "The reviews from this movie are not learnable."
    else:
        if not value:
            return ""
        else:
            te_vec = model_collect['inference'].infer(model_collect['inference'].process_text(value))
            score = model_collect['model'].predict(xgb.DMatrix(np.reshape(te_vec, (1, -1))))
            return "The input review is: {}. \n The predicted score for this movie is {}.".format(value, score)



# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/search':
        return page_1_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server()