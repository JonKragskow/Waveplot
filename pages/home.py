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
    '/orbitals',
    '/vibrations',
    '/f-densities',
]

names = [
    'Atomic Orbitals',
    'Harmonic Oscillators',
    '4f Densities',
]

images = [
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
