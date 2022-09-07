# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, dash_table, callback
from dash.dash_table import FormatTemplate
import colorlover
from collections import OrderedDict
import plotly.express as px
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

df = pd.read_csv('conf_LDAd.csv')
df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)  # TODO: make this line unnecessary
df.sort_values('year', inplace=True, ascending=False)
current_table = df


def discrete_background_color_bins(df, n_bins=5, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in df:
            df_numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = df.select_dtypes('number')
    else:
        df_numeric_columns = df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                            '{{{column}}} >= {min_bound}' +
                            (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return styles, html.Div(legend, style={'padding': '5px 0 5px 0'})


ldavis = open("ldavis.html", "r").read()
# print(ldavis)
app.layout = html.Div(children=[
    html.H1(children='General Conference Recommender'),

    # html.Div(children='''
    #     Dash: A web application framework for your data.
    # '''),

    # dcc.Graph(
    #     id='example-graph',
    #     figure=fig
    # ),

    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='perfect', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),
    html.Div(id='tbl_out'),

    html.H2(children="Recommended Talks:"),
    html.Div(id='recommendations'),
    html.H2(children="What are those percentages?"),
    html.Iframe(
        # enable all sandbox features
        # see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
        # this prevents javascript from running inside the iframe
        # and other things security reasons
        # sandbox='',
        srcDoc=ldavis,
        style={"height":900,"width":"100%"}
    )
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
)
def update_output_div(input_value):
    outputval = df[df['title'].str.contains(input_value, case=False)][:5]
    global current_table
    current_table = outputval
    outputval = outputval[['index', 'title', 'author', 'year']]
    return dash_table.DataTable(
        outputval.to_dict('records'),
        [{"name": i, "id": i} for i in outputval.columns],
        id='tbl',
        # hidden_columns=['index'],
        style_cell_conditional=[
            {'if': {'column_id': 'index', },
             'display': 'None', }]
    )


@callback(
    Output('tbl_out', 'children'),
    Output('recommendations', 'children'),
    Input('tbl', 'active_cell')
)
def update_graphs(active_cell):
    if (active_cell):
        resp = current_table.iloc[active_cell['row']]

        nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(df.loc[:, '0':])
        distances, indices = nbrs.kneighbors([df.loc[resp['index'], '0':]])

        rec = df.iloc[indices[0]]
        rec = rec[['index', 'title', 'author', 'year', 'month', '0', '1', '2', '3', '4', '5', '6', '7', '8']]

        # basic = FormatTemplate.Format()
        percentage = FormatTemplate.percentage(0, False)
        columns = [{"name": rec.columns[i], "id": rec.columns[i]} for i in range(5)]
        columns.extend([{"name": str(i+1), "id": str(i), 'type':'numeric', 'format':percentage} for i in range(9)])
        print(columns)
        # columns = [{"name": i, "id": i, 'type': 'numeric' if any(x.isdigit() for x in i) else 'any',
        #             'format': percentage if any(x.isdigit() for x in i) else basic} for i in rec.columns]

        (styles, legend) = discrete_background_color_bins(rec,
                                                          columns=['0', '1', '2', '3', '4', '5', '6', '7', '8'])

        return f"{resp['title']} by {resp['author']}, id:{resp['index']}", \
               [
                   legend,
                   dash_table.DataTable(
                       rec.to_dict('records'),
                       columns,
                       id='rec_table',
                       # hidden_columns=['index'],
                       style_cell_conditional=[
                           {'if': {'column_id': 'index', },
                            'display': 'None',
                            },
                           # {'if': {'column_type': 'text', },
                           #
                           #  },
                       ],
                       style_data_conditional=styles
                   )
               ]
    else:
        return "Click the table", "click the table above"


if __name__ == '__main__':
    app.run_server(debug=True)
