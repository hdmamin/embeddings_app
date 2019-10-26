import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go

from utils import Embeddings, empty_table, get_empty_table_data


CSS = ['https://codepen.io/Hmamin/pen/OJJgbBL.css']
app = dash.Dash(__name__, external_stylesheets=CSS)
app.config['suppress_callback_exceptions'] = True
server = app.server
emb = Embeddings.from_glove_file('/Users/hmamin/data/glove/glove.6B.50d.txt')
emb.save('emb.pkl')


###############################################################################
# App components
###############################################################################
div_similar = html.Div([
    html.H4('Input Word'),
    dcc.Input(id='input', size='23'),
    dcc.RadioItems(options=[{'label': x, 'value': x}
                             for x in ['euclidean', 'cosine']],
                   value='euclidean',
                   id='distance_selector'),
    html.H4('Similar Words'),
    empty_table(['Word', 'Distance'], 'output'),
    ], id='similar')

div_analogy = html.Div([html.H2('Analogies'),
                        html.Div([
                            dcc.Input(id='a'),
                            html.Div('is to'),
                            dcc.Input(id='b')
                        ], className='row'),
                        dcc.Input(id='c'),
                        empty_table(['Word'], id_='d')])

div_add = html.Div(html.H2('Arithmetic'))

div_detypo = html.Div(html.H2('Spell Check'))

div_cbow = html.Div(html.H2('Bag of Words'))

div_plot = html.Div([html.H2('2D Projection'),
                     # dcc.Dropdown(options=[{'label': k, 'value': k}
                     #                       for k in emb],
                     #              multi=True,
                     #              id='plot_selector'),
                     dcc.Markdown('Enter words in the text area below, '
                                  'separated by spaces. If a word doesn\'t '
                                  'show up on the chart, it means it\'s not '
                                  'present in our corpus.'),
                     dcc.Textarea(value='',
                                  style={'width': '100%'},
                                  id='plot_selector'),
                     html.Div(id='plot')
                     ])

###############################################################################
# Main tab layout
###############################################################################
app.layout = html.Div([html.H1('Fun With Embeddings'),
                       dcc.Tabs(id='tab_selector', value='similar',
                                children=[dcc.Tab(label='Similar Words',
                                                  value='similar'),
                                          dcc.Tab(label='Analogies',
                                                  value='analogy'),
                                          dcc.Tab(label='Arithmetic',
                                                  value='add'),
                                          dcc.Tab(label='Spell Check',
                                                  value='detypo'),
                                          dcc.Tab(label='CBOW',
                                                  value='cbow'),
                                          dcc.Tab(label='2D Projection',
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
                   detypo=div_detypo,
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
               Input('c', 'value')])
def update_analogy(a, b, c):
    print(a, b, c)
    if not (a and b and c):
        return get_empty_table_data(['Word'])

    data = emb.analogy(a, b, c)
    print(data)

    # Error handling for partially typed words or words not in vocab.
    if not data:
        return get_empty_table_data(['Word'])

    # Case where all words have been passed in.
    return [{'Word': word} for word in data.keys()]


@app.callback(Output('plot', 'children'),
              [Input('plot_selector', 'value')])
def update_plot(words):
    words = words.split()
    arr = np.array([emb.vec_2d(word) for word in words])
    print(arr)

    # TESTING 2
    # labels, arr = map(list, zip(*w2v.items()))
    # arr = np.array(arr)
    # print(labels)
    # print(arr)
    # TESTING

    # trace = go.Scatter(x=arr[:, 0],
    #                    y=arr[:, 1],
    #                    mode='markers',
    #                    marker=dict(size=12),
    #                    text=words)
    trace = [go.Scatter(x=[emb.vec_2d(word)[0]],
                        y=[emb.vec_2d(word)[1]],
                        mode='markers',
                        marker={'size': 12},
                        name=word,
                        text=word,
                        hoverinfo='x+y+text')
             for word in words
             if word in emb]
    layout = go.Layout(showlegend=True)
    fig = go.Figure(data=trace,
                    layout=layout)
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
