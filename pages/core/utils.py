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
import io
import base64
from dash.exceptions import PreventUpdate
import numpy as np
import numpy.linalg as la
import sys
from ase import neighborlist, Atoms
from ase.geometry.analysis import Analysis
import xyz_py as xyzp
import xyz_py.atomic as atomic


atom_colours = {
    "H": "#999999",
    "Li": "#9932CC",
    "Na": "#9932CC",
    "K": "#9932CC",
    "Rb": "#9932CC",
    "Cs": "#9932CC",
    "Fr": "#9932CC",
    "Be": "#4E7566",
    "Mg": "#4E7566",
    "Ca": "#4E7566",
    "Sr": "#4E7566",
    "Ba": "#4E7566",
    "Ra": "#4E7566",
    "Sc": "#4E7566",
    "Y": "#4E7566",
    "La": "#4E7566",
    "Ac": "#4E7566",
    "Ti": "#301934",
    "V": "#301934",
    "Cr": "#301934",
    "Mn": "#301934",
    "Fe": "#301934",
    "Ni": "#301934",
    "Co": "#301934",
    "Cu": "#301934",
    "Zn": "#301934",
    "Zr": "#301943",
    "Nb": "#301943",
    "Mo": "#301943",
    "Tc": "#301943",
    "Ru": "#301943",
    "Rh": "#301943",
    "Pd": "#301943",
    "Ag": "#301943",
    "Cd": "#301943",
    "Hf": "#301934",
    "Ta": "#301934",
    "W": "#301934",
    "Re": "#301934",
    "Os": "#301934",
    "Ir": "#301934",
    "Pt": "#301934",
    "Au": "#301934",
    "Hg": "#301934",
    "B": "#FFFF00",
    "C": "#696969",
    "N": "#0000FF",
    "O": "#FF0000",
    "F": "#228B22",
    "Al": "#800080",
    "Si": "#FF7F50",
    "P": "#FF00FF",
    "S": "#FFFF00",
    "Cl": "#228B22",
    "As": "#F75394",
    "Br": "#4A2600",
    "other": "#3f3f3f"
}

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


def footer(name="Jon Kragskow", site="https://www.kragskow.dev"):

    footer = html.Footer(
        className="footer_custom",
        children=[
            html.Div(
                [
                    dbc.Button(
                        name,
                        href=site,
                        style={
                            "color": "",
                            'backgroundColor': 'transparent',
                            'borderColor': 'transparent',
                            'boxShadow': 'none',
                            'textalign': 'top'
                        }
                    )
                ]
            )
        ],
        style={'whiteSpace': 'pre-wrap'}
    )

    return footer

