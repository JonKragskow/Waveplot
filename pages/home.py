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

from dash import html, register_page
import dash_bootstrap_components as dbc
from functools import reduce
from operator import mul

from .core import utils

PAGE_NAME = 'Home'
PAGE_PATH = '/'
PAGE_DESCRIPTION = 'An interactive wavefunction viewer by Jon Kragskow'

register_page(
    __name__,
    path=PAGE_PATH,
    name=PAGE_NAME,
    description=PAGE_DESCRIPTION
)

paths = [
    '/radial',
    '/orbitals',
    '/vibrations',
    '/f-densities',
]

names = [
    'Radial Functions',
    'Atomic Orbitals',
    'Harmonic Oscillators',
    '4f Densities',
]

images = [
    'assets/radial.png',
    'assets/Orbitals.png',
    'assets/Vibrations.png',
    'assets/4f_densities.png',
]

grid = []

for path, name, image in zip(paths, names, images):
    grid.append(
        dbc.Col(
            children=[
                html.A(
                    children=[
                        html.Img(
                            src=image,
                            style={
                                "width": "500px",
                                "height": "500px"
                            }
                        )
                    ],
                    href=path
                ),
                html.H4(name)
            ],
            class_name='col-6',
            style={
                "textAlign": "center"
            }
        )
    )

if len(grid) % 2:
    grid.append(dbc.Col([]))


def reshape(lst, shape):
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i*n:(i+1)*n], shape[1:]) for i in range(len(lst)//n)]


grid = reshape(grid, [2, 2])

layout = html.Div(
    [
        dbc.Row(
            gr
        ) for gr in grid
    ] + [utils.footer()],
    className="main_wrapper",
    style={"marginTop": "20px"}
)
