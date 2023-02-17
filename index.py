from dash import dcc, html, page_container, callback
from dash.dependencies import Output, Input
from app import app
from pages.core import utils

navbar = utils.create_navbar("/")

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='navdiv', children=[navbar]),
    page_container
])


# server=app.server
@callback(
    Output('navdiv', 'children'),
    [Input('url', 'pathname')]
)
def display_page(current_path):
    '''
    Update navbar to current page name using path
    '''
    navbar = utils.create_navbar(current_path)
    return navbar


if __name__ == '__main__':
    app.run_server(debug=True)
