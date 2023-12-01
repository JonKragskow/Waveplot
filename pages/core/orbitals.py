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
from dash import dcc, html, Input, Output, callback, no_update, \
    Patch
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from skimage import measure
import pandas as pd
from . import common
from . import radial as rc


def s_3d(n: int):
    '''
    Calculates s orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 1:
        zbound = 5.
        step = 0.1
    elif n == 2:
        zbound = 30.
        step = 1.
    elif n == 3:
        zbound = 40.
        step = 2.
    elif n == 4:
        zbound = 40.
        step = 2.
    elif n == 5:
        zbound = 60.
        step = 2.
    elif n == 6:
        zbound = 80.
        step = 2.

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_s(n, 2*r/n)

    ang = 0.5/np.sqrt(np.pi)

    wav = ang*rad

    spacing = [
        np.diff(np.arange(-zbound, zbound, step))[0],
        np.diff(np.arange(-zbound, zbound, step))[0],
        np.diff(np.arange(-zbound, zbound, step))[0],
    ]

    return wav, spacing


def p_3d(n: int):
    '''
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: ndarray of floats
        Meshgrid containing wavefunction
    '''

    if n == 2:
        zbound = 20
        step = 0.5
    elif n == 3:
        zbound = 30
        step = 0.5
    elif n == 4:
        zbound = 60
        step = 1.
    elif n == 5:
        zbound = 60
        step = 1.
    elif n == 6:
        zbound = 80
        step = 1.

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-0.75 * zbound, 0.75 * zbound, step),
        np.arange(-0.75 * zbound, 0.75 * zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    # radial wavefunction
    rad = rc.radial_p(n, 2 * r / n)

    # angular wavefunction
    ang = np.sqrt(3. / (4. * np.pi)) * x / r
    wav = ang * rad

    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    spacing = [
        np.diff(np.arange(-0.75 * zbound, 0.75 * zbound, step))[0],
        np.diff(np.arange(-0.75 * zbound, 0.75 * zbound, step))[0],
        np.diff(np.arange(-zbound, zbound, step))[0],
    ]

    return wav, spacing


def dz_3d(n: int):
    '''
    Calculates dz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 3:
        zbound = 70.
        step = 1.
    elif n == 4:
        zbound = 90.
        step = 2.
    elif n == 5:
        zbound = 110.
        step = 2.
    elif n == 6:
        zbound = 140.
        step = 2.

    x, y, z = np.meshgrid(
        np.arange(-0.9 * zbound, 0.9 * zbound, step),
        np.arange(-0.9 * zbound, 0.9 * zbound, step),
        np.arange(-zbound, zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2*r/n)

    ang = 2*z**2-x**2-y**2

    wav = rad*ang

    spacing = [
        np.diff(np.arange(-0.9 * zbound, 0.9 * zbound, step))[0],
        np.diff(np.arange(-0.9 * zbound, 0.9 * zbound, step))[0],
        np.diff(np.arange(-zbound, zbound, step))[0]
    ]

    return wav, spacing


def dxy_3d(n: int):
    '''
    Calculates dxy orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 3:
        zbound = 45.
        step = 1.
    elif n == 4:
        zbound = 70.
        step = 1.
    elif n == 5:
        zbound = 98.
        step = 2.
    elif n == 6:
        zbound = 135.
        step = 2.

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        copy=True
    )
    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2*r/n)

    ang = x*y

    wav = rad*ang

    spacing = np.diff(np.arange(-zbound, zbound, step))[0]

    spacing = [
        spacing, spacing, spacing
    ]

    return wav, spacing


def fz_3d(n: int):
    '''
    Calculates fz3 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        zbound = 100.
        step = 2.
    elif n == 5:
        zbound = 100.
        step = 2.
    elif n == 6:
        zbound = 130.
        step = 2.

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-0.75 * zbound, 0.75 * zbound, step),
        np.arange(-0.75 * zbound, 0.75 * zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)

    ang = 0.25 * np.sqrt(7/np.pi) * z*(2*z**2-3*x**2-3*y**2)/(r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad*ang

    spacing = [
        np.diff(np.arange(-0.75 * zbound, 0.75 * zbound, step))[0],
        np.diff(np.arange(-0.75 * zbound, 0.75 * zbound, step))[0],
        np.diff(np.arange(-zbound, zbound, step))[0],
    ]

    return wav, spacing


def fxyz_3d(n: int):
    '''
    Calculates fxyz orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        zbound = 60.
        step = 1.
    elif n == 5:
        zbound = 90.
        step = 2.
    elif n == 6:
        zbound = 115.
        step = 2.

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)

    ang = 0.5 * np.sqrt(105/np.pi) * x*y*z/(r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad*ang

    spacing = [step, step, step]

    return wav, spacing


def fyz2_3d(n: int):
    '''
    Calculates fyz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital

    Returns
    -------
    x: np.meshgrid
    y: np.meshgrid
    z: np.meshgrid
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        zbound = 65.
        step = 1.
    elif n == 5:
        zbound = 90.
        step = 2
    elif n == 6:
        zbound = 125.
        step = 2

    x, y, z = np.meshgrid(
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        np.arange(-zbound, zbound, step),
        copy=True
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)

    ang = 0.25 * np.sqrt(35/(2*np.pi)) * (3*x**2-y**2)*y/r**3
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad*ang

    spacing = [step, step, step]

    return wav, spacing


class PlotDiv(common.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.viewer = html.Div(
            id=self.prefix('mol_div'),
            className='molecule_div'
        )

        self.plot = dcc.Graph(
            id=self.prefix('plotly_iso'),
            className='plot_area',
            mathjax=True,
            figure={
                'data': [],
                'layout': {
                    'scene': {
                        'xaxis': {
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                            'showticklabels': False,
                            'visible': False
                        },
                        'yaxis': {
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                            'showticklabels': False,
                            'visible': False
                        },
                        'zaxis': {
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                            'showticklabels': False,
                            'visible': False
                        },
                        'aspectratio': dict(x=1., y=1, z=1.),
                        'dragmode': 'orbit'
                    }
                }
            },
            config=rc.BASIC_CONFIG
        )

        self.orb_store = dcc.Store(
            id=self.prefix('orbital_store'),
            data=[]
        )
        self.spacing_store = dcc.Store(
            id=self.prefix('spacing_store'),
            data=[]
        )
        self.make_div_contents()

    def make_div_contents(self):
        '''
        Assembles div children in rows and columns
        '''

        contents = [
            dbc.Row(
                [
                    dbc.Col([
                        dcc.Loading(self.plot)
                    ]),
                    self.orb_store,
                    self.spacing_store
                ]
            )
        ]

        self.div.children = contents
        return


class OptionsDiv(common.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.orb_select = dbc.Select(
            id=self.prefix('orb_name_3d'),
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
            ],
            value='3dz2',
            placeholder='Select an orbital'
        )

        self.x_input = dbc.Input(
            id=self.prefix('view_x'),
            value=0,
            type='number'
        )
        self.x_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'X'
                ),
                self.x_input
            ]
        )

        self.y_input = dbc.Input(
            id=self.prefix('view_y'),
            value=0,
            type='number'
        )
        self.y_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'Y'
                ),
                self.y_input
            ]
        )

        self.z_input = dbc.Input(
            id=self.prefix('view_z'),
            value=0,
            type='number'
        )
        self.z_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'Z'
                ),
                self.z_input
            ]
        )

        self.zoom_input = dbc.Input(
            id=self.prefix('view_zoom'),
            value='',
            type='number'
        )
        self.zoom_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'Zoom'
                ),
                self.zoom_input
            ]
        )

        self.qx_input = dbc.Input(
            id=self.prefix('view_qx'),
            value=0,
            type='number'
        )
        self.qx_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'qX'
                ),
                self.qx_input
            ]
        )

        self.qy_input = dbc.Input(
            id=self.prefix('view_qy'),
            value=0,
            type='number'
        )
        self.qy_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'qY'
                ),
                self.qy_input
            ]
        )

        self.qz_input = dbc.Input(
            id=self.prefix('view_qz'),
            value=0,
            type='number'
        )
        self.qz_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'qZ'
                ),
                self.qz_input
            ]
        )

        self.qw_input = dbc.Input(
            id=self.prefix('view_qw'),
            value=1,
            type='number'
        )
        self.qw_ig = self.make_input_group(
            [
                dbc.InputGroupText(
                    'qW'
                ),
                self.qw_input
            ]
        )

        self.cutaway_select = dbc.Select(
            id=self.prefix('cutaway_in'),
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            },
            options=[
                {
                    'label': 'None',
                    'value': 'none'
                },
                {
                    'label': '1/2 X',
                    'value': 'x'
                },
                {
                    'label': '1/2 Y',
                    'value': 'y'
                },
                {
                    'label': '1/2 Z',
                    'value': 'z'
                }
            ],
            value=1.
        )
        self.cutaway_ig = self.make_input_group(
            [
                dbc.InputGroupText('Cutaway'),
                self.cutaway_select
            ]
        )

        self.isoval_input = dbc.Input(
            id=self.prefix('isoval'),
            type='number',
            value=0.01,
            style={
                'text-align': 'center'
            },
            max=1.,
            min=0.0000000000001
        )

        self.isoval_ig = self.make_input_group(
            [
                dbc.InputGroupText('Isovalue'),
                self.isoval_input
            ]
        )

        self.colour_input_a = dbc.Input(
            id=self.prefix('colour_a'),
            type='color',
            value='#491688',
            style={
                'height': '40px'
            }
        )

        self.colour_input_b = dbc.Input(
            id=self.prefix('colour_b'),
            type='color',
            value='#ffeb0a',
            style={
                'height': '40px'
            }
        )

        self.colours_ig = self.make_input_group(
            [
                dbc.InputGroupText('Colours'),
                self.colour_input_a,
                self.colour_input_b
            ]
        )

        self.axes_check = dbc.Checkbox(
            value=False,
            id=self.prefix('axes')
        )

        self.axes_ig = self.make_input_group(
            [
                dbc.InputGroupText('Axes'),
                dbc.InputGroupText(
                    self.axes_check
                )
            ]
        )

        self.make_div_contents()
        return

    def make_input_group(self, elements):

        group = dbc.InputGroup(
            elements,
            className='mb-3 4 d-flex justify-content-center'
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
                            },
                        children='Options'
                    )
                )
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        self.orb_select,
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
                    )
                ]
            ),
            dbc.Row([
                dbc.Col([
                    self.cutaway_ig
                ]),
                dbc.Col(
                    self.isoval_ig
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    self.axes_ig,
                    className='mb-3'
                )
            ]),
            dbc.Row(
                dbc.Col(
                    html.H5(
                        style={'textAlign': 'center'},
                        children='Viewer'
                    )
                )
            ),
            dbc.Row([
                dbc.Col(
                    self.x_ig
                ),
                dbc.Col(
                    self.y_ig
                ),
                dbc.Col(
                    self.z_ig
                ),
                dbc.Col(
                    self.zoom_ig
                ),

            ]),
            dbc.Row([
                dbc.Col(
                    self.qx_ig
                ),
                dbc.Col(
                    self.qy_ig
                ),
                dbc.Col(
                    self.qz_ig
                ),
                dbc.Col(
                    self.qw_ig
                ),
            ])
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: PlotDiv, options_div: OptionsDiv):

    callback(
        [
            Output(plot_div.orb_store, 'data'),
            Output(plot_div.spacing_store, 'data')
        ],
        [
            Input(options_div.orb_select, 'value')
        ],
        prevent_initial_callback=True
    )(calc_wav)

    callback(
        [
            Output(plot_div.plot, 'figure')
        ],
        [
            Input(plot_div.orb_store, 'data'),
            Input(plot_div.spacing_store, 'data'),
            Input(options_div.cutaway_select, 'value'),
            Input(options_div.axes_check, 'value'),
            Input(options_div.isoval_input, 'value'),
            Input(options_div.colour_input_a, 'value'),
            Input(options_div.colour_input_b, 'value')
        ],
        prevent_initial_callback=True
    )(make_plotly_iso)


def calc_wav(orbital_name):

    if not orbital_name:
        return no_update

    n = int(orbital_name[0])
    name = orbital_name[1:]

    # Get orbital n value and name
    orb_func_dict = {
        's': s_3d,
        'p': p_3d,
        'dxy': dxy_3d,
        'dz2': dz_3d,
        'fxyz': fxyz_3d,
        'fyz2': fyz2_3d,
        'fz3': fz_3d
    }

    wav, spacing = orb_func_dict[name](
        n
    )
    wav = [wa.tolist() for wa in wav]

    return wav, spacing


def make_plotly_iso(wav, spacing, cutaway, axes_check, isoval, colour_1,
                    colour_2):

    if not len(wav):
        return no_update

    fig = Patch()

    if None in [cutaway, isoval, colour_1, colour_2]:
        return no_update

    colour_1 = colour_1.lstrip('#')
    colour_2 = colour_2.lstrip('#')

    colour_1 = tuple(int(colour_1[i:i+2], 16) for i in (0, 2, 4))
    colour_2 = tuple(int(colour_2[i:i+2], 16) for i in (0, 2, 4))

    rounds = 10
    try:
        verts1, faces1, _, _ = measure.marching_cubes(
            np.array(wav),
            isoval,
            spacing=spacing
        )
    except ValueError:
        return no_update
    verts1 = laplacian_smooth(verts1, faces1, rounds=rounds)
    x1, y1, z1 = verts1.T
    I1, J1, K1 = faces1.T

    try:
        verts2, faces2, _, _ = measure.marching_cubes(
            np.array(wav),
            -isoval,
            spacing=spacing
        )
    except ValueError:
        return no_update
    verts2 = laplacian_smooth(verts2, faces2, rounds=rounds)
    x2, y2, z2 = verts2.T
    I2, J2, K2 = faces2.T

    xzero = np.concatenate([x1, x2])
    yzero = np.concatenate([y1, y2])
    zzero = np.concatenate([z1, z2])
    x1 -= np.mean(xzero)
    x2 -= np.mean(xzero)
    y1 -= np.mean(yzero)
    y2 -= np.mean(yzero)
    z1 -= np.mean(zzero)
    z2 -= np.mean(zzero)

    trace2 = go.Mesh3d(
        x=x2,
        y=y2,
        z=z2,
        color='rgb({:d},{:d},{:d})'.format(*colour_2),
        i=I2,
        j=J2,
        k=K2,
        name='',
        showscale=False
    )
    trace1 = go.Mesh3d(
        x=x1,
        y=y1,
        z=z1,
        color='rgb({:d},{:d},{:d})'.format(*colour_1),
        i=I1,
        j=J1,
        k=K1,
        name='',
        showscale=False
    )
    traces = [trace1, trace2]
    if axes_check:
        trace_3 = go.Scatter3d(
            x=[-1.2 * np.max(x1), 1.2 * np.max(x1)],
            y=[0, 0],
            z=[0, 0],
            line={
                'color': 'blue',
                'width': 8
            },
            mode='lines',
            name='x'
        )
        trace_4 = go.Scatter3d(
            y=[-1.2 * np.max(y1), 1.2 * np.max(y1)],
            x=[0, 0],
            z=[0, 0],
            line={
                'color': 'green',
                'width': 8
            },
            mode='lines',
            name='y'
        )
        trace_5 = go.Scatter3d(
            z=[-1.2 * np.max(z1), 1.2 * np.max(z1)],
            x=[0, 0],
            y=[0, 0],
            line={
                'color': 'red',
                'width': 8
            },
            mode='lines',
            name='z'
        )
        traces += [trace_3, trace_4, trace_5]

    fig['data'] = traces

    if cutaway == 'x':
        fig['layout']['scene']['yaxis']['range'] = 'auto'
        fig['layout']['scene']['zaxis']['range'] = 'auto'
        fig['layout']['scene']['xaxis']['range'] = [
            np.min(np.concatenate([x1, x2])),
            0
        ]
        fig['layout']['scene']['aspectratio'] = {'x': 0.5, 'y': 1., 'z': 1.}
    elif cutaway == 'y':
        fig['layout']['scene']['zaxis']['range'] = 'auto'
        fig['layout']['scene']['xaxis']['range'] = 'auto'
        fig['layout']['scene']['yaxis']['range'] = [
            np.min(np.concatenate([y1, y2])),
            0
        ]
        fig['layout']['scene']['aspectratio'] = {'x': 1., 'y': 0.5, 'z': 1.}
    elif cutaway == 'z':
        fig['layout']['scene']['yaxis']['range'] = 'auto'
        fig['layout']['scene']['xaxis']['range'] = 'auto'
        fig['layout']['scene']['zaxis']['range'] = [
            np.min(np.concatenate([z1, z2])),
            0
        ]
        fig['layout']['scene']['aspectratio'] = {'x': 1., 'y': 1., 'z': 0.5}
    else:
        fig['layout']['scene']['xaxis']['range'] = 'auto'
        fig['layout']['scene']['yaxis']['range'] = 'auto'
        fig['layout']['scene']['zaxis']['range'] = 'auto'
        fig['layout']['scene']['aspectratio'] = {'x': 1., 'y': 1., 'z': 1.}

    return [fig]


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
