import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from utils import (Embeddings, empty_table, get_empty_table_data,
                   distance_selector)


CSS = ['https://codepen.io/Hmamin/pen/OJJgbBL.css']
app = dash.Dash(__name__, external_stylesheets=CSS)
app.config['suppress_callback_exceptions'] = True
server = app.server
# emb = Embeddings.from_glove_file('/Users/hmamin/data/glove/glove.6B.50d.txt')
emb = Embeddings.from_pickle('emb.pkl')

###############################################################################
# App components
###############################################################################
div_similar = html.Div([
    html.Div('Type a word in the input below. The table will show words with '
             'similar embeddings and their distance from your chosen word. '
             'You can try selecting different distance metrics to see how '
             'results vary.', className='row'),
    html.Div([
        html.H4('Input Word', className='three columns'),
        html.H4('Distance Metric', className='four columns')
    ], className='row '),
    html.Div([
        html.Div(
            dcc.Input(id='input', size='23'),
            className='three columns'
        ),
        html.Div(
            distance_selector('distance_selector'),
            className='four columns'
        )
    ], className='row'),
    html.H4('Similar Words', className='row'),
    empty_table(['Word', 'Distance'], 'output'),
], id='similar')


div_analogy = html.Div([
    html.Div([
        html.H4('Analogies', className='six columns'),
        html.H4('Distance Metric', className='three columns'),
    ], className='row'),
    html.Div([
        html.Div(
            dcc.Markdown(
                'We can use embeddings to fill in analogies of the form: '
                'A is to B as C is to _. You will enter words for A, B, and C '
                'in the spaces below. Candidates for the final word can be '
                'found by computing B-A+C and searching for nearest '
                'neighbors. Note that we always treat A and B as valid '
                'candidates to fill in the blank, while C is only considered '
                'as a candidate in the trivial case where A=B (in which case '
                'C should be the first choice). '
                '\n\nA few classic examples where the embeddings work '
                'reasonably well are "king is to queen as man is to \_" or '
                '"Paris is to France as Madrid is to \_". Results are mixed '
                'for more challenging analogies.'),
            className='six columns'),
        html.Div(
            distance_selector('analogy_distance_selector'),
            className='three columns'
        )
    ], className='row'),
    html.Div([
        html.Div(dcc.Input(debounce=True, id='a'), className='three columns'),
        html.H6('is to', className='two columns'),
        html.Div(dcc.Input(debounce=True, id='b', className='three columns')),
        html.H6('as', className='one column'),
        html.Div(dcc.Input(id='c'), className='two columns'),
    ], className='row'),
    html.H6('is to'),
    empty_table(['Word'], id_='d')])


div_add = html.Div([
    html.Div([
        html.H4('Arithmetic', className='six columns'),
        html.H4('Distance Metric', className='three columns')
    ], className='row'),
    html.Div([
        html.Div(),
        html.Div(distance_selector('add_distance_selector'),
                 className='three columns')
    ], className='row')
])


div_cbow = html.Div([
    html.Div([
        html.H4('Bag of Words', className='six columns'),
        html.H4('Distance Metric', className='three columns'),
    ], className='row'),
    html.Div([
        html.Div(
        dcc.Markdown('Enter words in the text area below, separated by '
                     'spaces. Hit *Enter* to submit. This will compute the '
                     'mean embedding of all input words and search for the '
                     'word whose embedding most closely matches this average. '
                     'While averaging over a bag of words in this manner can '
                     'be somewhat effective at tasks like identifying similar '
                     'documents, my brief experiments here looking for single '
                     'words near the average embedding did not yield '
                     'particularly promising results. Perhaps with some '
                     'adjustments, something useful might emerge.'),
            className='six columns'),
        html.Div(distance_selector('cbow_distance_selector'),
                 className='three columns')
    ], className='row'),
    dcc.Textarea(value='',
                 style={'width': '100%'},
                 id='cbow_selector'),
    empty_table(['Word'], id_='cbow_table')
])


div_plot = html.Div([
    html.H4('2D Projection'),
    dcc.Markdown('Type one or more words in the text area below. Hitting the '
                 '`enter` key will update the chart (you can do this after '
                 'each word, or type multiple space-separated words and '
                 'submit at the end). If a word doesn\'t show up on the plot, '
                 'it means it\'s not present in our embedding vocabulary.'),
    dcc.Textarea(value='scientist engineer developer statistician\n',
                 style={'width': '100%'},
                 id='plot_selector'),
    dcc.Graph(figure=go.Figure(data=[],
                               layout=go.Layout(showlegend=True)),
              id='plot'),
    html.Div('Note: these axes don\'t correspond to any particular dimension '
             'that we can interpret. They are simply the result of using PCA '
             'to reduce the embedding dimensionality to 2.')
])

###############################################################################
# Main tab layout
###############################################################################
app.layout = html.Div([html.H1('Fun With Embeddings'),
                       dcc.Tabs(id='tab_selector', value='similar',
                                children=[dcc.Tab(label='Neighbors',
                                                  value='similar'),
                                          dcc.Tab(label='Analogies',
                                                  value='analogy'),
                                          dcc.Tab(label='Arithmetic',
                                                  value='add'),
                                          dcc.Tab(label='Bag of Words',
                                                  value='cbow'),
                                          dcc.Tab(label='Plot',
                                                  value='plot')]),
                       html.Div(id='content_div')],
                      className='container')

###############################################################################
# App callbacks
###############################################################################
@app.callback(Output('content_div', 'children'),
             [Input('tab_selector', 'value')])
def render_tab(tab):
    tab2div = dict(similar=div_similar,
                   analogy=div_analogy,
                   add=div_add,
                   cbow=div_cbow,
                   plot=div_plot)
    return tab2div[tab]


@app.callback(Output('output', 'data'),
              [Input('input', 'value'),
               Input('distance_selector', 'value')])
def update_similar(word, distance):
    if not word or word not in emb:
        return [{'Word': '', 'Distance': ''} for i in range(5)]

    # Case where word is in vocab.
    neighbors = emb.nearest_neighbors(word=word, distance=distance)
    return [{'Word': k, 'Distance': v} for k, v in neighbors.items()]


@app.callback(Output('d', 'data'),
              [Input('a', 'value'),
               Input('b', 'value'),
               Input('c', 'value'),
               Input('analogy_distance_selector', 'value')])
def update_analogy(a, b, c, distance):
    print(a, b, c)
    if not (a and b and c):
        return get_empty_table_data(['Word'])

    words = emb.analogy(a, b, c, distance=distance)
    print(words)

    # Error handling for partially typed words or words not in vocab.
    if not words:
        return get_empty_table_data(['Word'])

    # Case where all words have been passed in.
    return [{'Word': word} for word in words]


@app.callback(Output('plot', 'figure'),
              [Input('plot_selector', 'value')])
def update_plot(words):
    # Only add traces when user submits a new word, otherwise callbacks will
    # overlap and PCA will be run repeatedly.
    # A new trace is used for each word so the legend can label each point.
    traces = []
    if words.endswith('\n'):
        for word in words.split():
            vec = emb.vec_2d(word)
            print('vec', vec)
            if vec is None:
                continue

            # Confirmed word in vocab so we add a new trace to the list.
            trace = go.Scatter(x=[vec[0]],
                               y=[vec[1]],
                               mode='markers',
                               marker={'size': 12},
                               name=word,
                               text=word,
                               hoverinfo='x+y+text')
            traces.append(trace)

    return go.Figure(data=traces,
                     layout=go.Layout(showlegend=True))


@app.callback(Output('cbow_table', 'data'),
              [Input('cbow_selector', 'value'),
               Input('cbow_distance_selector', 'value')])
def update_cbow(words, distance):
    if not words.endswith('\n'):
        return get_empty_table_data(['word'])

    data = emb.cbow_neighbors(*words.split(), distance=distance)
    print('UPDATE_CBOW data:', data, '\n\n\n\n')
    return [{'Word': word} for word in data.keys()]


if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
