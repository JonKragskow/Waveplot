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
import numpy as np
from dash import html, Input, Output, callback, no_update, \
    Patch, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from skimage import measure
import pandas as pd
import uuid
import copy

from . import common as com
from . import radial as rc


ORB_CONFIG = copy.deepcopy(com.BASIC_CONFIG)
ORB_CONFIG['toImageButtonOptions']['format'] = 'png'
ORB_CONFIG['toImageButtonOptions']['scale'] = 2
ORB_CONFIG['toImageButtonOptions']['filename'] = 'orbital'
ORB_CONFIG['toImageButtonOptions']['width'] = None
ORB_CONFIG['toImageButtonOptions']['height'] = None

ORB_LAYOUT = copy.deepcopy(com.BASIC_LAYOUT)
ORB_LAYOUT['xaxis']['showline'] = False
ORB_LAYOUT['xaxis']['ticks'] = ''
ORB_LAYOUT['xaxis']['showticklabels'] = False
ORB_LAYOUT['xaxis']['minor_ticks'] = ''
ORB_LAYOUT['yaxis']['showline'] = False
ORB_LAYOUT['yaxis']['ticks'] = ''
ORB_LAYOUT['yaxis']['showticklabels'] = False
ORB_LAYOUT['yaxis']['minor_ticks'] = ''
ORB_LAYOUT['scene'] = copy.deepcopy(com.BASIC_SCENE)
ORB_LAYOUT['margin'] = dict(r=20, b=10, l=10, t=10)
ORB_LAYOUT['showlegend'] = False
ORB_LAYOUT['uirevision'] = 'same'
ORB_LAYOUT['scene']['uirevision'] = 'same'

DEFAULT_ISO = {
    '1s': 0.1,
    '2s': 0.01,
    '3s': 0.001,
    '4s': 0.001,
    '5s': 0.001,
    '6s': 0.0005,
    '2p': 0.01,
    '3p': 0.001,
    '4p': 0.001,
    '5p': 0.001,
    '6p': 0.0006,
    '3dz2': 0.1,
    '4dz2': 0.1,
    '5dz2': 0.1,
    '6dz2': 0.1,
    '3dxy': 0.01,
    '4dxy': 0.01,
    '5dxy': 0.01,
    '6dxy': 0.01,
    '4fz3': 0.0005,
    '5fz3': 0.0005,
    '6fz3': 0.0004,
    '4fxyz': 0.0006,
    '5fxyz': 0.0006,
    '6fxyz': 0.0004,
    '4fyz2': 0.0006,
    '5fyz2': 0.0006,
    '6fyz2': 0.0006,
    'sp': 0.01,
    'sp2': 0.01,
    'sp3': 0.01
}


def s_3d(n: int, half: str = ''):
    '''
    Calculates s orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 1:
        bound = 5.
        step = 0.1
    elif n == 2:
        bound = 30.
        step = 1.
    elif n == 3:
        bound = 40.
        step = 2.
    elif n == 4:
        bound = 40.
        step = 2.
    elif n == 5:
        bound = 60.
        step = 2.
    elif n == 6:
        bound = 80.
        step = 2.

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_s(n, 2 * r / n)

    ang = 0.5 / np.sqrt(np.pi)

    wav = ang * rad

    return wav


def p_3d(n: int, half: str = ''):
    '''
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: ndarray of floats
        Meshgrid containing wavefunction
    '''

    if n == 2:
        bound = 20
        step = 0.5
    elif n == 3:
        bound = 30
        step = 0.5
    elif n == 4:
        bound = 60
        step = 1.
    elif n == 5:
        bound = 60
        step = 1.
    elif n == 6:
        bound = 80
        step = 1.

    x = np.arange(-bound, bound, step)
    y = np.arange(-0.75 * bound, 0.75 * bound, step)
    z = np.arange(-0.75 * bound, 0.75 * bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, 0.75 * bound, step)
    elif half == 'z':
        z = np.arange(0, 0.75 * bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    # radial wavefunction
    rad = rc.radial_p(n, 2 * r / n)

    # angular wavefunction
    ang = np.sqrt(3. / (4. * np.pi)) * x / r
    wav = ang * rad

    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    return wav


def dz_3d(n: int, half: str = ''):
    '''
    Calculates dz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 3:
        bound = 70.
        step = 1.
    elif n == 4:
        bound = 90.
        step = 2.
    elif n == 5:
        bound = 110.
        step = 2.
    elif n == 6:
        bound = 140.
        step = 2.

    x = np.arange(-0.95 * bound, 0.95 * bound, step)
    y = np.arange(-0.95 * bound, 0.95 * bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, 0.95 * bound, step)
    elif half == 'y':
        y = np.arange(0, 0.95 * bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2 * r / n)

    ang = 2 * z**2 - x**2 - y**2

    wav = rad * ang

    return wav


def dxy_3d(n: int, half: str = ''):
    '''
    Calculates dxy orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 3:
        bound = 45.
        step = 1.
    elif n == 4:
        bound = 70.
        step = 1.
    elif n == 5:
        bound = 98.
        step = 2.
    elif n == 6:
        bound = 135.
        step = 2.

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )
    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2 * r / n)

    ang = x * y

    wav = rad * ang

    return wav


def fz_3d(n: int, half: str = ''):
    '''
    Calculates fz3 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 4:
        bound = 100.
        step = 2.
    elif n == 5:
        bound = 100.
        step = 2.
    elif n == 6:
        bound = 130.
        step = 2.

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )
    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2 * r / n)

    ang = 0.25 * np.sqrt(7 / np.pi) * z * (2 * z**2 - 3 * x**2 - 3 * y**2) / (r**3) # noqa
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad * ang

    return wav


def sp_3d(half: str):

    bound = 20
    step = 0.5

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    # radial wavefunction
    radp = rc.radial_p(2, r)

    # angular wavefunction
    angp = np.sqrt(3. / (4. * np.pi)) * x / r
    wavp = angp * radp

    rads = rc.radial_s(2, r)
    angs = 0.5 / np.sqrt(np.pi)
    wavs = angs * rads

    wav = 1. / np.sqrt(2) * wavs + 1. / np.sqrt(2) * wavp
    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    return wav


def sp3_3d(half: str):

    bound = 20
    step = 0.5

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    # radial wavefunction
    radp = rc.radial_p(2, r)

    # angular wavefunction
    angp1 = np.sqrt(3. / (4. * np.pi)) * x / r
    wavp1 = angp1 * radp

    rads = rc.radial_s(2, r)
    angs = 0.5 / np.sqrt(np.pi)
    wavs = angs * rads

    wav = 0.5 * wavs + np.sqrt(3) / 2. * wavp1
    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    return wav


def sp2_3d(half: str):

    bound = 20
    step = 0.5

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    # radial wavefunction
    radp = rc.radial_p(2, r)

    # angular wavefunction
    angp1 = np.sqrt(3. / (4. * np.pi)) * x / r
    wavp1 = angp1 * radp

    angp2 = np.sqrt(3. / (4. * np.pi)) * y / r
    wavp2 = angp2 * radp

    rads = rc.radial_s(2, r)
    angs = 0.5 / np.sqrt(np.pi)
    wavs = angs * rads

    wav = 1. / np.sqrt(3) * wavs + np.sqrt(2/3) * wavp1 + 1. / np.sqrt(2) * wavp2 # noqa
    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    return wav


def fxyz_3d(n: int, half: str = ''):
    '''
    Calculates fxyz orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 4:
        bound = 60.
        step = 1.
    elif n == 5:
        bound = 90.
        step = 2.
    elif n == 6:
        bound = 115.
        step = 2.

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2 * r / n)

    ang = 0.5 * np.sqrt(105 / np.pi) * x * y * z / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fyz2_3d(n: int, half: str = ''):
    '''
    Calculates fyz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        If non-empty, specified axis will go from 0 tobound

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    '''

    if n == 4:
        bound = 65.
        step = 1.
    elif n == 5:
        bound = 90.
        step = 2
    elif n == 6:
        bound = 125.
        step = 2

    x = np.arange(-bound, bound, step)
    y = np.arange(-bound, bound, step)
    z = np.arange(-bound, bound, step)

    if half == 'x':
        x = np.arange(0, bound, step)
    elif half == 'y':
        y = np.arange(0, bound, step)
    elif half == 'z':
        z = np.arange(0, bound, step)

    x, y, z = np.meshgrid(
        x,
        y,
        z,
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2 * r / n)

    ang = 0.25 * np.sqrt(35 / (2 * np.pi)) * (3 * x**2 - y**2) * y / r**3
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad * ang

    return wav


class OptionsDiv(com.Div):
    def __init__(self, prefix, default_orb='3dz2', **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.orb_select = dbc.Select(
            id=self.prefix('orbital_select'),
            style={
                'textAlign': 'center',
                'width': '50%',
                'align': 'center'
            },
            options=[
                {'label': '1s', 'value': '1s'},
                {'label': '2s', 'value': '2s'},
                {'label': '3s', 'value': '3s'},
                {'label': '4s', 'value': '4s'},
                {'label': '5s', 'value': '5s'},
                {'label': '6s', 'value': '6s'},
                {'label': '2p', 'value': '2p'},
                {'label': '3p', 'value': '3p'},
                {'label': '4p', 'value': '4p'},
                {'label': '5p', 'value': '5p'},
                {'label': '6p', 'value': '6p'},
                {'label': '3dz²', 'value': '3dz2'},
                {'label': '4dz²', 'value': '4dz2'},
                {'label': '5dz²', 'value': '5dz2'},
                {'label': '6dz²', 'value': '6dz2'},
                {'label': '3dxy', 'value': '3dxy'},
                {'label': '4dxy', 'value': '4dxy'},
                {'label': '5dxy', 'value': '5dxy'},
                {'label': '6dxy', 'value': '6dxy'},
                {'label': '4fz³', 'value': '4fz3'},
                {'label': '5fz³', 'value': '5fz3'},
                {'label': '6fz³', 'value': '6fz3'},
                {'label': '4fxyz', 'value': '4fxyz'},
                {'label': '5fxyz', 'value': '5fxyz'},
                {'label': '6fxyz', 'value': '6fxyz'},
                {'label': '4fyz²', 'value': '4fyz2'},
                {'label': '5fyz²', 'value': '5fyz2'},
                {'label': '6fyz²', 'value': '6fyz2'},
                {'label': 'sp', 'value': 'sp'},
                {'label': 'sp²', 'value': 'sp2'},
                {'label': 'sp³', 'value': 'sp3'}
            ],
            value=default_orb,
            placeholder='Select an orbital'
        )

        self.orb_ig = self.make_input_group(
            [
                dbc.InputGroupText('Orbital'),
                self.orb_select
            ]
        )

        self.half_select = dbc.Select(
            id=str(uuid.uuid1()),
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            },
            options=[
                {
                    'label': 'None',
                    'value': 'full'
                },
                {
                    'label': 'xz plane',
                    'value': 'x'
                },
                {
                    'label': 'yz plane',
                    'value': 'y'
                },
                {
                    'label': 'xy plane',
                    'value': 'z'
                }
            ],
            value='full'
        )
        self.half_ig = self.make_input_group(
            [
                dbc.InputGroupText('Cut through'),
                self.half_select
            ]
        )

        self.isoval_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='number',
            value=0.01,
            style={
                'text-align': 'center'
            },
            max=1.,
            min=0.0000000000001
        )

        self.update_isoval_btn = dbc.Button(
            children='Default'
        )

        self.isoval_ig = self.make_input_group(
            [
                dbc.InputGroupText('Isovalue'),
                self.isoval_input,
                self.update_isoval_btn
            ]
        )

        self.colour_input_a = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#0000ff',
            style={
                'height': '40px'
            }
        )

        self.colour_input_b = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#ffff00',
            style={
                'height': '40px'
            }
        )

        self.colours_ig = self.make_input_group(
            [
                dbc.InputGroupText('Orbital colours'),
                self.colour_input_a,
                self.colour_input_b
            ]
        )

        self.axes_check = dbc.Checkbox(
            value=True,
            id=str(uuid.uuid1()),
            style={'margin-left': '5%'}
        )

        self.x_axis_col_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#000000',
            style={
                'height': '40px'
            }
        )

        self.y_axis_col_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#000000',
            style={
                'height': '40px'
            }
        )

        self.z_axis_col_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#000000',
            style={
                'height': '40px'
            }
        )

        self.axes_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    ['Show axes', self.axes_check]
                ),
                self.x_axis_col_input,
                self.y_axis_col_input,
                self.z_axis_col_input
            ]
        )

        self.make_div_contents()
        return

    def make_input_group(self, elements):

        group = dbc.InputGroup(
            elements,
            className='mb-3'
        )
        return group

    def make_div_contents(self):
        '''
        Assembles div children in rows and columns
        '''

        contents = [
            dbc.Row([
                dbc.Col(
                    html.H4(
                        style={
                            'textAlign': 'center',
                            'margin-bottom': '5%',
                            'margin-top': '5%'
                        },
                        children='Configuration'
                    )
                )
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        self.orb_ig,
                        className='4 d-flex justify-content-center mb-3',
                        style={'align': 'center'}
                    )
                ]
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        self.colours_ig,
                        className='mb-3'
                    ),
                    dbc.Col(
                        self.axes_ig,
                        className='mb-3'
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([
                        self.half_ig
                    ]),
                    dbc.Col(
                        self.isoval_ig
                    )
                ]
            )
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: com.PlotDiv, options_div: OptionsDiv):

    # Catch orbital name and update isovalue display
    callback(
        [
            Output(options_div.isoval_input, 'value', allow_duplicate=True),
            Output(plot_div.store, 'data')
        ],
        Input(options_div.orb_select, 'value'),
        prevent_initial_call=True
    )(lambda x: (DEFAULT_ISO[x], x))

    # Suggest new isovalue from list of "good" values
    callback(
        Output(options_div.isoval_input, 'value', allow_duplicate=True),
        Input(options_div.update_isoval_btn, 'n_clicks'),
        State(options_div.orb_select, 'value'),
        prevent_initial_call=True
    )(lambda x, y: DEFAULT_ISO[y])

    callback(
        [
            Output(options_div.x_axis_col_input, 'disabled'),
            Output(options_div.y_axis_col_input, 'disabled'),
            Output(options_div.z_axis_col_input, 'disabled')
        ],
        Input(options_div.axes_check, 'value')
    )(lambda x: [not x] * 3)

    callback(
        Output(plot_div.plot, 'figure', allow_duplicate=True),
        [
            Input(plot_div.store, 'data'),
            Input(options_div.axes_check, 'value'),
            Input(options_div.isoval_input, 'value'),
            Input(options_div.half_select, 'value')
        ],
        [
            State(options_div.x_axis_col_input, 'value'),
            State(options_div.y_axis_col_input, 'value'),
            State(options_div.z_axis_col_input, 'value'),
            State(options_div.colour_input_a, 'value'),
            State(options_div.colour_input_b, 'value')
        ],
        prevent_initial_call='initial_duplicate'
    )(plot_data)

    callback(
        Output(plot_div.plot, 'figure', allow_duplicate=True),
        [
            Input(options_div.colour_input_a, 'value'),
            Input(options_div.colour_input_b, 'value'),
            Input(options_div.x_axis_col_input, 'value'),
            Input(options_div.y_axis_col_input, 'value'),
            Input(options_div.z_axis_col_input, 'value')
        ],
        prevent_initial_call=True
    )(update_iso_colour)

    return


def calc_wav(orbital_name, half=''):
    '''
    Calculates given wavefunction's 3d data and saves to assets folder
    '''

    # Get orbital n value and name
    orb_func_dict = {
        's': s_3d,
        'p': p_3d,
        'dxy': dxy_3d,
        'dz2': dz_3d,
        'fxyz': fxyz_3d,
        'fyz2': fyz2_3d,
        'fz3': fz_3d,
        'sp': sp_3d,
        'sp2': sp2_3d,
        'sp3': sp3_3d
    }

    special = ['sp', 'sp2', 'sp3']

    if orbital_name not in special:

        n = int(orbital_name[0])
        name = orbital_name[1:]
        wav = orb_func_dict[name](
            n, half
        )
    else:
        wav = orb_func_dict[orbital_name](half)

    name = f'assets/{half}{orbital_name}'
    np.save(name, wav)

    return


def plot_data(orb_name: str, axes_check: bool, isoval: float, half: str,
              x_col: str, y_col: str, z_col: str, colour_1: str,
              colour_2: str):
    '''
    Finds isosurface for given wavefunction data using marching cubes,
    then smooths the surface and plots as mesh
    '''

    if None in [isoval, colour_1, colour_2, x_col, y_col, z_col]:
        return no_update

    if half == 'full':
        half = ''

    wav = np.load(f'assets/{half}{orb_name}.npy')

    # Calculate each isosurface and smooth it
    rounds = 3
    try:
        verts1, faces1, _, _ = measure.marching_cubes(
            np.array(wav),
            level=isoval
        )
    except ValueError:
        return no_update
    verts1 = laplacian_smooth(verts1, faces1, rounds=rounds)
    x1, y1, z1 = verts1.T
    I1, J1, K1 = faces1.T

    try:
        verts2, faces2, _, _ = measure.marching_cubes(
            np.array(wav),
            -isoval
        )
    except ValueError:
        if orb_name == '1s':
            verts2 = copy.deepcopy(verts1)
            faces2 = copy.deepcopy(faces1)
        else:
            return no_update
    verts2 = laplacian_smooth(verts2, faces2, rounds=rounds)
    x2, y2, z2 = verts2.T
    I2, J2, K2 = faces2.T

    # Shift surface origin to zero
    if half != 'y':
        xzero = np.concatenate([x1, x2])
        x1 -= np.mean(xzero)
        x2 -= np.mean(xzero)

    if half != 'x':
        yzero = np.concatenate([y1, y2])
        y1 -= np.mean(yzero)
        y2 -= np.mean(yzero)

    if half != 'z':
        zzero = np.concatenate([z1, z2])
        z1 -= np.mean(zzero)
        z2 -= np.mean(zzero)

    # Make mesh of each isosurface
    trace1 = go.Mesh3d(
        x=x1,
        y=y1,
        z=z1,
        color=colour_1,
        i=I1,
        j=J1,
        k=K1,
        name='',
        showscale=False
    )
    if orb_name != '1s':
        trace2 = go.Mesh3d(
            x=x2,
            y=y2,
            z=z2,
            color=colour_2,
            i=I2,
            j=J2,
            k=K2,
            name='',
            showscale=False
        )
        traces = [trace1, trace2]
    else:
        traces = [trace1]

    # Add axes
    lim = 1.5 * np.max(np.concatenate([x1, x2, y1, y2, z1, z2]))

    # Add invisible scatter plot to preserve aspect ratio
    # since Patching layout.scene breaks the save button's orientation...
    if not axes_check:
        traces.append(
            go.Scatter3d(
                x=[-lim, lim, 0, 0, 0, 0,],
                y=[0, 0, -lim, lim, 0, 0],
                z=[0, 0, 0, 0, -lim, lim],
                mode='markers',
                marker={
                    'color': 'rgba(255, 255, 255, 0.)'
                },
                hoverinfo='skip'
            )
        )
    trace_3 = go.Scatter3d(
        x=[-lim, lim],
        y=[0, 0],
        z=[0, 0],
        line={
            'color': x_col,
            'width': 8
        },
        mode='lines',
        hoverinfo='text',
        hovertext='x'
    )
    trace_4 = go.Scatter3d(
        y=[-lim, lim],
        x=[0, 0],
        z=[0, 0],
        line={
            'color': y_col,
            'width': 8
        },
        mode='lines',
        hoverinfo='text',
        hovertext='y'
    )
    trace_5 = go.Scatter3d(
        z=[-lim, lim],
        x=[0, 0],
        y=[0, 0],
        line={
            'color': z_col,
            'width': 8
        },
        mode='lines',
        hoverinfo='text',
        hovertext='z'
    )
    if axes_check:
        traces += [trace_3, trace_4, trace_5]

    fig = Patch()

    fig['data'] = traces

    return fig


def update_iso_colour(colour_1, colour_2, x_col, y_col, z_col):

    fig = Patch()

    fig['data'][0]['color'] = colour_1
    fig['data'][1]['color'] = colour_2
    fig['data'][2]['line']['color'] = x_col
    fig['data'][3]['line']['color'] = y_col
    fig['data'][4]['line']['color'] = z_col

    return fig


def laplacian_smooth(vertices, faces, rounds=1):
    '''
    Pure-python reference implementation of laplacian smoothing.
    Smooth the mesh in-place.
    This is simplest mesh smoothing technique, known as Laplacian Smoothing.
    Relocates each vertex by averaging its position with those of its adjacent
    neighbors.
    Repeat for N iterations.
    One disadvantage of this technique is that it results in overall shrinkage
    of the mesh, especially for many iterations. (But nearly all smoothing
    techniques
    cause at least some shrinkage.)
    (Obviously, any normals you have for this mesh will be out-of-date after
    smoothing.)

    Parameters
    ----------
    vertices:
        Vertex coordinates shape=(N,3)
    faces
        Face definitions.  shape=(N,3)
        Each row lists 3 vertices (indexes into the vertices array)
    rounds:
        How many passes to take over the data.
        More iterations results in a smoother mesh, but more shrinkage
        (and more CPU time).
    Returns:
        new vertices
    '''
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces)

    # Compute the list of all unique vertex adjacencies
    all_edges = np.concatenate([faces[:, (0, 1)],
                                faces[:, (1, 2)],
                                faces[:, (2, 0)]])
    all_edges.sort(axis=1)
    edges_df = pd.DataFrame(all_edges, columns=['v1_id', 'v2_id'])
    edges_df.drop_duplicates(inplace=True)
    del all_edges

    # (This sort isn't technically necessary, but it might give
    # better cache locality for the vertex lookups below.)
    edges_df.sort_values(['v1_id', 'v2_id'], inplace=True)

    # How many neighbors for each vertex
    # i.e. how many times it is mentioned in the edge list
    neighbor_counts = np.bincount(
        edges_df.values.reshape(-1),
        minlength=len(vertices)
    )

    new_vertices = np.empty_like(vertices)
    for _ in range(rounds):
        new_vertices[:] = vertices

        # For the complete edge index list, accumulate (sum) the vertexes on
        # the right side of the list into the left side's address and vversa
        #
        # We want something like this:
        # v1_indexes, v2_indexes = df['v1_id'], df['v2_id']
        # new_vertices[v1_indexes] += vertices[v2_indexes]
        # new_vertices[v2_indexes] += vertices[v1_indexes]
        #
        # ...but that doesn't work because v1_indexes will contain repeats,
        #    and "fancy indexing" behavior is undefined in that case.
        #
        # Instead, it turns out that np.ufunc.at() works (it's an "unbuffered"
        # operation)
        np.add.at(
            new_vertices, edges_df['v1_id'], vertices[edges_df['v2_id'], :]
        )
        np.add.at(
            new_vertices, edges_df['v2_id'], vertices[edges_df['v1_id'], :]
        )

        # (plus one here because each point itself is also included in the sum)
        new_vertices[:] /= (neighbor_counts[:, None] + 1)

        # Swap (save RAM allocation overhead by reusing the new_vertices array
        # between iterations)
        vertices, new_vertices = new_vertices, vertices

    return vertices
