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
import plotly.colors as pc
import dash_bootstrap_components as dbc
from dash import html, page_registry

tol_cols = [
    "rgb(0  , 0  , 0)",
    "rgb(230, 159, 0)",
    "rgb(86 , 180, 233)",
    "rgb(0  , 158, 115)",
    "rgb(240, 228, 66)",
    "rgb(0  , 114, 178)",
    "rgb(213, 94 , 0)",
    "rgb(204, 121, 167)"
]
# Bang wong list of colourblindness friendly colours
# https://www.nature.com/articles/nmeth.1618
wong_cols = [
    "rgb(51 , 34 , 136)",
    "rgb(17 , 119, 51)",
    "rgb(68 , 170, 153)",
    "rgb(136, 204, 238)",
    "rgb(221, 204, 119)",
    "rgb(204, 102, 119)",
    "rgb(170, 68 , 153)",
    "rgb(136, 34 , 85)"
]
# Default list of colours is plotly"s safe colourlist
def_cols = pc.qualitative.Safe


def dash_id(page: str):
    def func(_id: str):
        return f"{page}_{_id}"
    return func


def create_navbar(current_path):
    """
    Creates navbar element for current_page

    Parameters
    ----------
    current_page : str
        Name of webpage which navbar will appear on

    Returns
    -------
    dbc.NavbarSimple
        Navbar focussed on current page, with other pages included
    """

    paths = [
        page['path'] for page in page_registry.values()
    ]

    names = [
        page['name'] for page in page_registry.values()
    ]

    current_name = None
    for path, name in zip(paths, names):
        if current_path == path:
            current_name = name

    dropdown = [dbc.DropdownMenuItem("More pages", header=True)]

    for path, name in zip(paths, names):
        if path not in [current_path, '/']:
            dropdown.append(
                dbc.DropdownMenuItem(
                    name,
                    href=path
                )
            )

    # Icons for navbar
    # these are hyperlinked
    icons = dbc.Row(
        [
            dbc.Col(
                html.A(
                    html.I(
                        className="fab fa-github fa-2x",
                        style={
                            'color': 'white'
                        }
                    ),
                    href="https://github.com/jonkragskow/waveplot"
                )
            ),
            dbc.Col(
                html.A(
                    html.I(
                        className="fa-solid fa-question fa-2x",
                        style={
                            'color': 'white'
                        }
                    ),
                    href="/assets/waveplot_docs.pdf",
                )
            ),
            # dbc.Col(
            #     html.A(
            #         html.I(
            #             className="fa-solid fa-book fa-2x",
            #             style={
            #                 'color': 'white'
            #             }
            #         ),
            #         href="PAPER_URL"
            #     )
            # )
        ],
        style={
            'position': 'absolute',
            'right': '40px'
        },
        class_name='nav_buttons'
    )

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(current_name, href=current_path)),
            dbc.DropdownMenu(
                children=dropdown,
                nav=True,
                in_navbar=True,
                label="More",
            ),
            html.Div(icons)
        ],
        brand="Waveplot",
        brand_href="/",
        color="#307cff",
        dark=True,
        links_left=True,
    )
    return navbar
