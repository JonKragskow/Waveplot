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
    clientside_callback, ClientsideFunction, Patch, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from . import common
from . import radial as rc


def s_3d(n: int, cutaway: float = 1.):
    '''
    Calculates s orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 1:
        upper = 10.
        step = 2*upper/50.
        lower = - upper
    elif n == 2:
        upper = 17.
        step = 2*upper/50.
        lower = - upper
    elif n == 3:
        upper = 30.
        step = 2*upper/50.
        lower = - upper
    elif n == 4:
        upper = 45.
        step = 2*upper/50.
        lower = - upper
    elif n == 5:
        upper = 58.
        step = 2*upper/60.
        lower = - upper
    elif n == 6:
        upper = 75.
        step = 2*upper/70.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_s(n, 2*r/n)

    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.5/np.sqrt(np.pi)

    wav = ang*rad

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, 0.0005, step


def p_3d(n: int, cutaway: float = 1.):
    '''
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 1:
        upper = 10.
        step = 2*upper/50.
        lower = - upper
    elif n == 2:
        upper = 17.
        step = 2*upper/50.
        lower = - upper
    elif n == 3:
        upper = 30.
        step = 2*upper/50.
        lower = - upper
    elif n == 4:
        upper = 45.
        step = 2*upper/50.
        lower = - upper
    elif n == 5:
        upper = 60.
        step = 2*upper/60.
        lower = - upper
    elif n == 6:
        upper = 80.
        step = 2*upper/70.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_p(n, 2*r/n)

    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = np.sqrt(3./(4.*np.pi)) * z/r

    wav = ang*rad

    if n == 1:
        ival = 0.0005
    if n == 2:
        ival = 0.0005
    if n == 3:
        ival = 0.0005
    elif n == 4:
        ival = 0.0005
    elif n == 5:
        ival = 0.0005
    elif n == 6:
        ival = 0.0010

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, ival, step


def dz_3d(n: int, cutaway: float = 1.):
    '''
    Calculates dz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 3:
        upper = 50.
        step = 2*upper/60.
        lower = - upper
    elif n == 4:
        upper = 70.
        step = 2*upper/70.
        lower = - upper
    elif n == 5:
        upper = 98.
        step = 2*upper/80.
        lower = - upper
    elif n == 6:
        upper = 135.
        step = 2*upper/90.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 2*z**2-x**2-y**2

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, 0.08, step


def dxy_3d(n: int, cutaway: float = 1.):
    '''
    Calculates dxy orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 3:
        upper = 45.
        step = 2*upper/60.
        lower = - upper
    elif n == 4:
        upper = 70.
        step = 2*upper/70.
        lower = - upper
    elif n == 5:
        upper = 98.
        step = 2*upper/80.
        lower = - upper
    elif n == 6:
        upper = 135.
        step = 2*upper/90.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_d(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = x*y

    wav = rad*ang

    if n == 3:
        ival = 0.005
    elif n == 4:
        ival = 0.01
    elif n == 5:
        ival = 0.01
    elif n == 6:
        ival = 0.01

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, ival, step


def fz_3d(n: int, cutaway: float = 1.):
    '''
    Calculates fz3 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        upper = 100.
        step = 2*upper/70.
        lower = - upper
    elif n == 5:
        upper = 100.
        step = 2*upper/70.
        lower = - upper
    elif n == 6:
        upper = 130.
        step = 2*upper/85.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(7/np.pi) * z*(2*z**2-3*x**2-3*y**2)/(r**3)

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, 0.000005, step


def fxyz_3d(n: int, cutaway: float = 1.):
    '''
    Calculates fxyz orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        upper = 60.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 90.
        step = 2*upper/70.
        lower = - upper
    elif n == 6:
        upper = 115.
        step = 2*upper/80.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.5 * np.sqrt(105/np.pi) * x*y*z/(r**3)

    wav = rad*ang

    if n == 4:
        ival = 0.000005
    elif n == 5:
        ival = 0.000005
    elif n == 6:
        ival = 0.000005

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, ival, step


def fyz2_3d(n: int, cutaway: float = 1.):
    '''
    Calculates fyz2 orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    cutaway: int
        number used to split orbital in half

    Returns
    -------
    x: np.mgrid
        x values
    y: np.mgrid
        y values
    z: np.mgrid
        z values
    wav: np.mgrid
        wavefunction values at x, y, z
    upper: float
        max value of axes
    lower: float
        min value of axes
    ival: float
        isoval for orbital plotting
    '''

    if n == 4:
        upper = 65.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 90.
        step = 2*upper/90.
        lower = - upper
    elif n == 6:
        upper = 125.
        step = 2*upper/100.
        lower = - upper

    x, y, z = np.mgrid[
        lower:upper:step,
        lower:upper:step,
        lower:upper:step
    ]

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = rc.radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(35/(2*np.pi)) * (3*x**2-y**2)*y/r**3

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, x, y, z, 0.000005, step


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
                        },
                        'yaxis': {
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                        },
                        'zaxis': {
                            'showgrid': False,
                            'zeroline': False,
                            'showline': False,
                        },
                        'aspectratio': dict(x=1., y=1, z=1.)
                    }
                }
            },
            config=rc.BASIC_CONFIG
        )

        self.orb_store = dcc.Store(
            id=self.prefix('orbital_store'),
            data=''
        )
        self.isoval_store = dcc.Store(
            id=self.prefix('isoval_store'),
            data=0
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
                        # self.viewer,
                        self.plot
                    ]),
                    self.orb_store,
                    self.isoval_store
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
                'width': '50%'
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
            value='',
            placeholder='Select an orbital'
        )

        self.download_button = dbc.Button(
            'Download Image',
            id=self.prefix('download_image'),
            style={
                'boxShadow': 'none',
                'textalign': 'top'
            },
            className='me-1',
        )

        self.download_hidden_div = html.Div(
            id=self.prefix('download-hdiv'),
            style={'display': 'none'}
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
                    'value': 1.
                },
                {
                    'label': '1/2',
                    'value': 0.5
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
            value='0.1',
            style={
                'height': '40px'
            },
            step=0.001,
            max=1.,
            min=0.00000000001
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

        self.wireframe_check = dbc.Checkbox(
            value=False,
            id=self.prefix('wireframe')
        )

        self.wireframe_ig = self.make_input_group(
            [
                dbc.InputGroupText('Wireframe'),
                dbc.InputGroupText(
                    self.wireframe_check
                )
            ]
        )

        self.make_div_contents()
        return

    def make_input_group(self, elements):

        group = dbc.InputGroup(
            elements,
            className='mb-3',
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
                        children='Orbital'
                    )
                )
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        self.orb_select,
                        className='mb-3'
                    )
                ],
                className='align-items-center'
            ),
            html.H4(
                style={'textAlign': 'center'},
                children='Plot Options'
            ),
            dbc.Row(
                dbc.Col(
                    html.H5(
                        style={'textAlign': 'center'},
                        children='Viewer',
                        className='mb-3'
                    )
                ),
                className='mb-3'
            ),
            dbc.Row([
                dbc.Col(
                    [
                        self.download_button,
                        self.download_hidden_div
                    ],
                    className='mb-3',
                    style={'textAlign': 'center'}
                )
                ],
                className='align-items-center'
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
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4(
                        style={
                            'textAlign': 'center',
                        },
                        children='Plot Options'
                    )
                ])
            ]),
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
                    self.wireframe_ig,
                    className='mb-3'
                ),
                dbc.Col(
                    self.isoval_ig,
                    className='mb-3'
                )
            ])
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: PlotDiv, options_div: OptionsDiv):

    callback(
        [
            Output(plot_div.plot, 'figure')
        ],
        [
            Input(options_div.orb_select, 'value'),
            Input(options_div.cutaway_select, 'value'),
            Input(options_div.isoval_input, 'value'),
            Input(options_div.colour_input_a, 'value'),
            Input(options_div.colour_input_b, 'value')
        ],
        prevent_initial_callback=True
    )(make_plotly_iso)

    # # Clientside callback for javascript molecule viewer
    # clientside_callback(
    #     '''
    #     function (dummy) {

    #         let canvas = document.getElementById('viewer_canvas');
    #         if (canvas == null){
    #             return;
    #         }
    #         var duri = canvas.toDataURL('image/png', 1)
    #         downloadURI(duri, 'orbital.png');

    #         return ;
    #         }
    #     ''', # noqa
    #     Output(options_div.download_hidden_div, 'children'),
    #     [
    #         Input(options_div.download_button, 'n_clicks'),
    #     ],
    #     prevent_initial_call=True
    # )

    # # Viewer callback
    # clientside_callback(
    #     ClientsideFunction(
    #         namespace='clientside',
    #         function_name='orbital_function'
    #     ),
    #     [
    #         Output(options_div.zoom_input, 'value'),
    #     ],
    #     [
    #         Input(plot_div.orb_store, 'data'),
    #         Input(plot_div.isoval_store, 'data'),
    #         Input(options_div.colour_input_a, 'value'),
    #         Input(options_div.colour_input_b, 'value'),
    #         Input(options_div.wireframe_check, 'value'),
    #         Input(options_div.orb_select, 'value'),
    #         Input(options_div.x_input, 'value'),
    #         Input(options_div.y_input, 'value'),
    #         Input(options_div.z_input, 'value'),
    #         Input(options_div.zoom_input, 'value'),
    #         Input(options_div.qx_input, 'value'),
    #         Input(options_div.qy_input, 'value'),
    #         Input(options_div.qz_input, 'value'),
    #         Input(options_div.qw_input, 'value')
    #     ],
    #     prevent_initial_call=True
    # )

    return


def make_plotly_iso(orbital_name, cutaway, isoval, colour_1, colour_2):

    fig = Patch()
    if None in [orbital_name, cutaway, isoval, colour_1, colour_2]:
        return no_update

    if isinstance(ctx.triggered_id, str) and 'iso' in ctx.triggered_id:
        fig['data'][0]['isomin'] = -isoval
        fig['data'][0]['isomax'] = isoval
        return [fig]

    colour_1 = colour_1.lstrip('#')
    colour_2 = colour_2.lstrip('#')

    colour_1 = tuple(int(colour_1[i:i+2], 16) for i in (0, 2, 4))
    colour_2 = tuple(int(colour_2[i:i+2], 16) for i in (0, 2, 4))

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

    n_points, wav, x, y, z, _, step = orb_func_dict[name](
        n
    )

    traces = go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=wav.flatten(),
        isomin=-float(isoval),
        isomax=float(isoval),
        caps=dict(
            x_show=False, y_show=False, z_show=False
        ),
        surface_count=2,
        colorscale=[
            [0, 'rgb({:d},{:d},{:d})'.format(*colour_1)],
            [1, 'rgb({:d},{:d},{:d})'.format(*colour_2)]
        ],
        showlegend=False
    )

    fig['data'] = [traces]

    if float(cutaway) == 0.5:
        fig['layout']['scene']['xaxis']['range'] = [np.min(x), 0]
        fig['layout']['scene']['aspectratio'] = {'x': 0.5, 'y': 1., 'z': 1.}
    else:
        fig['layout']['scene']['xaxis']['range'] = 'auto'
        fig['layout']['scene']['aspectratio'] = {'x': 1., 'y': 1., 'z': 1.}

    return [fig]
