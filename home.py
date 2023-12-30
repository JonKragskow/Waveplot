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

PAGE_NAME = 'Home'
PAGE_PATH = '/'
PAGE_DESCRIPTION = 'An interactive wavefunction viewer by Jon Kragskow'


cols = [
    dbc.Col(
        children=[
            html.A(
                children=[
                    html.Img(
                        src=item['image'],
                        style={
                            'aspect-ratio': '1 / 1',
                            'max-height': '40vh'
                        }
                    )
                ],
                href=item['path']
            ),
            html.H4(
                item['name']
            )
        ],
        sm=12,
        lg=3,
        style={
            'textAlign': 'center'
        },
        class_name='mb-3'
    )
    for name, item in page_registry.items()
    if name != PAGE_NAME
]

layout = dbc.Container(
    children=[
        html.Div(style={'height': '10%'}),
        dbc.Row(
            cols,
            align='center',
            style={
                'height': '80%',
                'padding-bottom': '100px'
            }
        ),
        html.Div(style={'height': '10%'}),
        com.make_footer()
    ],
    fluid=True,
    style={
        'margin-top': '20px',
        'height': '90vh',
        'overflow': 'scroll'
    }
)
