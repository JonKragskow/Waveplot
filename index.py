"""
                    Waveplot: An online wavefunction viewer
                    Copyright (C) 2023  Jon G. C. Kragskow

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from dash import dcc, html, page_container, callback, register_page
from dash.dependencies import Output, Input
from app import app
from pages.core import utils
import home

navbar = utils.create_navbar("/")

page_container.style = {
    'height': '100%'
}

app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(id='navdiv', children=[navbar]),
        page_container
    ],
    style={'height': '80vh'}
)


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


register_page(
    'pages',
    path=home.PAGE_PATH,
    name=home.PAGE_NAME,
    description=home.PAGE_DESCRIPTION,
    layout=home.layout
)


if __name__ == '__main__':
    app.run(debug=True)
