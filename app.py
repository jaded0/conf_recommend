# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, dash_table, callback
import plotly.express as px
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from crontab import CronTab

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

cron = CronTab(user='app')
for job in cron:
    print(job)
job = cron.new('python scraper.py')
job.day.every(1)
cron.write()

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

df = pd.read_csv('conf_LDAd.csv')
df.rename(columns={'Unnamed: 0':'index'}, inplace=True) # TODO: make this line unnecessary
current_table = df

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
    html.Div(id='recommendations')
])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value'),
)
def update_output_div(input_value):
    outputval = df[df['title'].str.contains(input_value, case=False)][:5]
    global current_table
    current_table = outputval
    outputval = outputval[['index','title', 'author', 'year', ]]
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
        rec = rec[['index', 'title', 'author', 'year', 'month', 'topics']]

        return f"{resp['title']} by {resp['author']}, id:{resp['index']}", \
               dash_table.DataTable(
                   rec.to_dict('records'),
                   [{"name": i, "id": i} for i in rec.columns],
                   id='rec_table',
                   # hidden_columns=['index'],
                   style_cell_conditional=[
                       {'if': {'column_id': 'index', },
                        'display': 'None', }]
               )
    else: return "Click the table","click the table"


if __name__ == '__main__':
    app.run_server(debug=True)
