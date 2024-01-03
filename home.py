'''
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
'''

from dash import html, page_registry
import dash_bootstrap_components as dbc

from pages.core import common as com

PAGE_NAME = 'Waveplot'
PAGE_PATH = '/'
PAGE_DESCRIPTION = 'An interactive wavefunction viewer by Jon Kragskow'


grid = [
    html.Div(
        children=[
            html.A(
                children=[
                    html.Img(
                        src=item['image'],
                        style={
                            'width': '100%',
                            'aspectRatio': '1/1'
                        }
                    )
                ],
                href=item['path']
            ),
            html.H4(item['name'], style={'textAlign': 'center'})
        ],
        className='card'
    )
    for name, item in page_registry.items()
    if name != PAGE_NAME
]

layout = dbc.Container(
    [html.Div(grid, className='cards'), com.make_footer()],
    style={
        'marginTop': '20px',
        'paddingBottom': '50px',
        'height': '100%',
        'overflow': 'auto'
    }
)
