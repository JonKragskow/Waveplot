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

from dash import html, dcc, callback_context, register_page, callback, \
    clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import io

from .core import orbitals as orbs
from .core import utils
from .core import common

ID_PREFIX = 'orb3_'
dash_id = common.dash_id(ID_PREFIX)

PAGE_NAME = '3d Orbitals'
PAGE_PATH = '/orbitals3d'
PAGE_IMAGE = 'assets/Orbitals3.png'
PAGE_DESCRIPTION = 'Interactive atomic orbitals'

# register_page(
#     __name__,
#     order=1,
#     path=PAGE_PATH,
#     name=PAGE_NAME,
#     title=PAGE_NAME,
#     image=PAGE_IMAGE,
#     description=PAGE_DESCRIPTION
# )

prefix = utils.dash_id('orb3')


class Orb3dPlot(common.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.viewer = html.Div(
            self.prefix('orb_mol_div'),
            className='molecule_div'
        )

        self.orb_store = dcc.Store(id=self.prefix('orbital_store'))
        self.isoval_store = dcc.Store(id=self.prefix('isoval_store'))

        self.make_div_contents()

    def make_div_contents(self):
        '''
        Assembles div children in rows and columns
        '''

        contents = [
            dbc.Row(
                [
                    dbc.Col(
                        self.viewer
                    ),
                    self.orb_store,
                    self.isoval_store
                ]
            )
        ]

        self.div.children = contents
        return

