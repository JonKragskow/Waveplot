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
from numpy.typing import NDArray
from dash import dcc, html, Input, Output, State, callback, no_update, Patch
import dash_bootstrap_components as dbc
from . import common as com
import scipy.constants as con
import io
import plotly.graph_objects as go
import copy
import uuid

# c in cm s-1
LIGHT = con.speed_of_light * 100
HBAR = con.hbar
H = con.Planck
NA = con.Avogadro

PIAB_LAYOUT = copy.deepcopy(com.BASIC_LAYOUT)
PIAB_LAYOUT.yaxis.title = {
    'text': 'E (10<sup>-19</sup>J)',
    'font': {
        'family': 'Arial',
        'size': 14,
        'color': 'black'
    }
}
PIAB_LAYOUT.xaxis.autorange = False
PIAB_LAYOUT.xaxis.range = [-0.125, 1.125]
PIAB_LAYOUT.xaxis.tickmode = 'array'
PIAB_LAYOUT.xaxis.tickvals = [0, 0.25, 0.5, 0.75, 1]
PIAB_LAYOUT.xaxis.ticktext = ['0', 'L/4', 'L/2', '3L/4', 'L']
PIAB_LAYOUT.showlegend = False
PIAB_LAYOUT.shapes = [
    dict(
        type='line',
        xref='x',
        yref='paper',
        x0=1, y0=0, x1=1, y1=1,
        line_color='black',
    ),
    dict(
        type='rect',
        xref='x',
        yref='paper',
        x0=1, y0=0, x1=1.125, y1=1,
        fillcolor='gray',
        line_color='gray'
    ),
    dict(
        type='line',
        xref='x',
        yref='paper',
        x0=0, y0=0, x1=0, y1=1,
        line_color='black',
    ),
    dict(
        type='rect',
        xref='x',
        yref='paper',
        x0=-0.125, y0=0, x1=0, y1=1,
        fillcolor='gray',
        line_color='gray'
    ),
]

PIAB_CONFIG = copy.deepcopy(com.BASIC_CONFIG)
PIAB_CONFIG['toImageButtonOptions']['format'] = 'png'
PIAB_CONFIG['toImageButtonOptions']['scale'] = 2
PIAB_CONFIG['toImageButtonOptions']['filename'] = 'particle_in_a_box'


def calc_piab_energies(length: float, mass: float, max_n: int) -> NDArray:
    '''
    Calculate energies of particle in a 1d box

    Parameters
    ----------
    length: float
        Length of box in nanometres
    mass: float
        Mass of particle in kg
    max_n: int
        Maximum value of n used for states
    Returns
    -------
    ndarray of floats
        Harmonic state energies for quantum oscillator in J
    '''
    # Convert length to metres
    length *= 1E-9

    # Particle in a box energies
    energies = np.array(
        [
            (H**2 * n**2) / (8 * mass * length**2)
            for n in range(1, max_n + 1)
        ]
    )

    return energies


def calc_piab_wf(length: float, max_n: int) -> NDArray:
    '''
    Calculates particle in a 1d box wavefunctions

    Parameters
    ----------
    length: float
        Length of box in nanometres
    max_n: int
        Maximum value of n used for states
    Returns
    -------
    ndarray of floats
        Particle in a box wavefunctions
    '''

    # Convert length to metres
    length *= 1E-9

    x = np.linspace(0, length, 1000)

    # Wavefunctions
    wf = np.array([
        np.sqrt(2 / length) * np.sin(x * n * np.pi / length)
        for n in range(1, max_n + 1)
    ])

    wf = np.array([
        w / np.max(wf)
        for w in wf
    ])

    return wf


class OptionsDiv(com.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.mass_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=9.1093837E-31,
            type='number',
            min=1E-50,
            value=9.1093837E-31,
            style={
                'textAlign': 'center'
            }
        )
        self.mass_ig = self.make_input_group(
            [
                dbc.InputGroupText('Particle Mass'),
                self.mass_input,
                dbc.InputGroupText(r'kg')
            ]
        )
        self.mass_ig.id = self.prefix('mass_input')

        self.length_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=0.12,
            type='number',
            min=0.000000001,
            value=0.12,
            style={
                'textAlign': 'center'
            }
        )
        self.length_ig = self.make_input_group(
            [
                dbc.InputGroupText('Box Length'),
                self.length_input,
                dbc.InputGroupText(r'nm')
            ]
        )

        self.max_n_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=5,
            value=5,
            type='number',
            min=1,
            max=100,
            style={
                'textAlign': 'center'
            }
        )

        self.max_n_ig = self.make_input_group(
            [
                dbc.InputGroupText('Max. n'),
                self.max_n_input
            ]
        )

        self.wf_scale_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=50,
            value=50,
            type='number',
            min=0,
            max=1E9,
            style={
                'textAlign': 'center'
            }
        )

        self.wf_scale_ig = self.make_input_group(
            [
                dbc.InputGroupText('WF Scale'),
                self.wf_scale_input
            ]
        )

        self.wf_toggle_check = dbc.Checkbox(
            value=True,
            id=str(uuid.uuid1()),
        )
        self.wf_toggle_ig = self.make_input_group(
            [
                dbc.InputGroupText('WF On/Off'),
                dbc.InputGroupText(
                    self.wf_toggle_check
                )
            ]
        )

        self.wf_pos_colour_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#1f77b4',
            style={
                'height': '40px'
            }
        )

        self.wf_neg_colour_input = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#d62728',
            style={
                'height': '40px'
            }
        )

        self.wf_ftype_select = dbc.Select(
            options=[
                {
                    'label': 'Probability',
                    'value': 'psi2'
                },
                {
                    'label': 'Wavefunction',
                    'value': 'psi'
                }
            ],
            value='psi'
        )

        self.wf_ftype_ig = self.make_input_group(
            [
                dbc.InputGroupText('Plot:'),
                self.wf_ftype_select
            ]
        )

        self.wf_colour_ig = self.make_input_group(
            [
                dbc.InputGroupText('Colours'),
                self.wf_pos_colour_input,
                self.wf_neg_colour_input
            ]
        )

        self.linewidth_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=2.,
            type='number',
            min=1,
            max=10,
            value=2.,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.linewidth_ig = self.make_input_group(
            [
                dbc.InputGroupText('Linewidth'),
                self.linewidth_input
            ]
        )

        self.download_data_btn = dbc.Button(
            'Download Data',
            id=str(uuid.uuid1()),
            style={
                'boxShadow': 'none',
                'width': '100%'
            }
        )
        self.download_data_tr = dcc.Download(
            id=str(uuid.uuid1()),
        )

        self.image_format_select = dbc.Select(
            id=str(uuid.uuid1()),
            style={
                'textAlign': 'center',
                'horizontalAlign': 'center',
                'display': 'inline'
            },
            options=[
                {
                    'label': 'svg',
                    'value': 'svg',
                },
                {
                    'label': 'png',
                    'value': 'png',
                },
                {
                    'label': 'jpeg',
                    'value': 'jpeg',
                }
            ],
            value='svg'
        )

        self.image_format_ig = self.make_input_group(
            [
                dbc.InputGroupText('Image format'),
                self.image_format_select
            ]
        )

        self.make_div_contents()
        return

    def make_input_group(self, elements):

        group = dbc.InputGroup(
            id=str(uuid.uuid1()),
            children=elements,
            class_name='mb-3',
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
                        id=self.prefix('parameters_header'),
                        style={
                            'textAlign': 'center',
                            'marginBottom': '5%',
                            'marginTop': '5%'
                        },
                        children='Configuration'
                    )
                )
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            self.mass_ig,
                            dbc.Tooltip(
                                children='Default is electron mass',
                                target=self.mass_ig.id
                            )
                        ],
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        [
                            self.length_ig,
                            dbc.Tooltip(
                                children='Default is ethene C-C distance',
                                target=self.length_ig.id
                            )
                        ],
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        [
                            self.max_n_ig
                        ],
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row([
                dbc.Col(
                    [
                        self.wf_colour_ig,
                    ],
                    class_name='mb-3'
                ),
                dbc.Col(
                    [
                        self.linewidth_ig,
                    ],
                    class_name='mb-3'
                )
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        self.wf_ftype_ig
                    ]
                ),
                dbc.Col(
                    [
                        self.wf_scale_ig,
                        dbc.Tooltip(
                            children='Scale wavefunction size (cosmetic)',
                            target=self.wf_scale_ig.id
                        )
                    ]
                ),
                dbc.Col(
                    [
                        self.wf_toggle_ig
                    ]
                )
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4(
                        style={
                            'textAlign': 'center',
                            'marginBottom': '5%',
                            'marginTop': '5%'
                        },
                        children='Output'
                    )
                ])
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            self.image_format_ig,
                            dbc.Tooltip(
                                children='Use the camera button in the top right of the plot to save an image', # noqa
                                target=self.image_format_ig.id
                            )
                        ],
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        [
                            self.download_data_btn,
                            self.download_data_tr
                        ],
                        class_name='mb-3'
                    )
                ]
            )
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: com.PlotDiv, options: OptionsDiv):

    # Calculate data and store
    inputs = [
        Input(options.length_input, 'value'),
        Input(options.mass_input, 'value'),
        Input(options.max_n_input, 'value')
    ]
    callback(
        Output(plot_div.store, 'data'),
        inputs
    )(calc_data)

    # Plot data
    inputs = [
        Input(plot_div.store, 'data'),
        Input(options.wf_pos_colour_input, 'value'),
        Input(options.wf_neg_colour_input, 'value'),
        Input(options.linewidth_input, 'value'),
        Input(options.wf_scale_input, 'value'),
        Input(options.wf_toggle_check, 'value'),
        Input(options.wf_ftype_select, 'value')
    ]

    callback(
        Output(plot_div.plot, 'figure', allow_duplicate=True),
        inputs,
        prevent_initial_call='initial_duplicate'
    )(update_plot)

    # Interaction with WF, PE, and state modals
    # Save data to file
    inputs = [
        Input(options.download_data_btn, 'n_clicks')
    ]
    states = [
        State(plot_div.store, 'data')
    ]
    callback(
        Output(options.download_data_tr, 'data'),
        inputs, states, prevent_initial_call=True
    )(download_data)

    # Update plot save format
    outputs = [
        Output(plot_div.plot, 'figure', allow_duplicate=True),
        Output(plot_div.plot, 'config')
    ]
    callback(
        outputs,
        Input(options.image_format_select, 'value'),
        prevent_initial_call=True
    )(update_save_format)

    return


def calc_data(length: float, mass: float, max_n: int) -> dict[str: list]:
    '''
    Calculate energies and wavefunctions of particle in a 1d box

    Parameters
    ----------
    length: float
        Length of box in nanometres
    mass: float
        Mass of particle in kg
    max_n: int
        Maximum value of n used for states

    Returns
    -------
    data: dict[str: list]
        Keys and values:\n
        'x': list[float] x values in m\n
        'energies': list[float] Energies in Joules\n
        'wf': list[list[float]] Wavefunction
    '''

    if None in [length, mass]:
        return no_update

    if max_n is None:
        max_n = 5

    final = {
        'x': np.linspace(0, length, 1000),
        'energies': calc_piab_energies(length, mass, max_n).tolist(),
        'wf': calc_piab_wf(length, max_n).tolist(),
    }

    return final


def update_plot(data: dict[str, list], pcolour_wf: str, ncolour_wf: str,
                lw: float, wf_scale: float, wf_toggle: bool,
                wf_prob: str) -> Patch:
    '''
    Plots harmonic state energies and wavefunctions, and harmonic potential

    Parameters
    ----------
    data: dict[str: list]
        Keys and values:\n
        'x': list[float] x values in m\n
        'energies': list[float] Energies in Joules\n
        'wf': list[list[float]] Wavefunction
    pcolour_wf: str
        Colour of positive wavefunction parts
    ncolour_wf: str
        Colour of negative wavefunction parts
    lw: float
        Linewidth of wavefunction and energy levels
    wf_scale: float
        Scale factor for wavefunction
    wf_toggle: bool
        If true, wavefunction is plotted
    wf_prob: str {'psi', 'psi2'}
        Which function to plot, either wavefunction or wavefunction^2
    Returns
    -------
    Patch
        Patch graph figure
    '''

    if None in [lw, wf_scale]:
        return no_update

    traces = []

    # Plot harmonic wavefunction and states

    _states = [
        [
            10**19 * (da),
            10**19 * (da)
        ]
        for da in data['energies']
    ]

    _x = [
        0, 1
    ]
    x_vals = np.linspace(0, 1, 1000)

    if wf_prob == 'psi2':
        wf_scale *= 2

    for nit, (wf, state) in enumerate(zip(np.asarray(data['wf']), _states)):
        traces.append(
            go.Scatter(
                x=_x,
                y=state,
                line={
                    'color': 'rgba(0,0,0,0)'
                },
                mode='lines',
                hoverinfo='skip'
            )
        )
        if wf_prob == 'psi2':
            wf *= wf
        if wf_toggle:
            # Find split point for +ve and negative
            wf_posi = np.where(wf >= 0.)[0]
            wf_negi = np.where(wf < 0.)[0]

            # Plot positive values
            # Set negative values to 0 to avoid plotly fill bug
            poswf = copy.copy(wf)
            poswf[wf_negi] = 0.
            traces.append(
                go.Scatter(
                    x=x_vals,
                    y=poswf * wf_scale + state[0],
                    line={
                        'color': pcolour_wf,
                        'width': lw
                    },
                    connectgaps=False,
                    fill='tonexty',
                    hoverinfo='skip'
                )
            )
            # Add invisible traces for next tonexty fill
            traces.append(
                go.Scatter(
                    x=_x,
                    y=state,
                    line={
                        'color': 'rgba(0,0,0,0)'
                    },
                    mode='lines',
                    hoverinfo='skip'
                )
            )
            # Plot negative values
            # Set positive values to 0 to avoid plotly fill bug
            negwf = copy.copy(wf)
            negwf[wf_posi] = 0.
            traces.append(
                go.Scatter(
                    x=x_vals,
                    y=negwf * wf_scale + state[0],
                    line={
                        'color': ncolour_wf,
                        'width': lw
                    },
                    connectgaps=False,
                    fill='tonexty',
                    hoverinfo='skip'
                )
            )
        traces.append(
            go.Scatter(
                x=x_vals,
                y=[state[0]] * len(x_vals),
                line={
                    'color': 'black',
                    'width': lw
                },
                mode='lines',
                hoverinfo='text',
                hovertext=f'n = {nit + 1:d}, E={state[0]:.0f} x 10<sup>-19</sup> J' # noqa
            )
        )

    fig = Patch()
    fig['data'] = traces

    return fig


def download_data(_nc: int, data: dict) -> dict:
    '''
    Creates output file for particle in a box wavefunctions and energies

    Parameters
    ----------
    _nc: int
        Number of clicks on download button. Used only to trigger callback
    data: dict[str: list]
        Keys and values:\n
        'x': list[float] x values in nm\n
        'energies': list[float] Energies in Joules\n
        'wf': list[list[float]] Wavefunction
    Returns
    -------
    dict
        Output dictionary used by dcc.Download
    '''

    if not len(data):
        return no_update

    oc = io.StringIO()

    header = (
        'Particle in a box wavefunction data calculated using waveplot.com\n'
        'A tool by Jon Kragskow\n'
    )

    oc.write(header)

    oc.write('n, Energies (J)\n')
    for it, energy in enumerate(data['energies']):
        oc.write(f'{it + 1:d}, {energy:.8e}\n')

    oc.write(
        '\nDisplacement (nm), Wavefunction for n=1, n=2, ...\n'
    )

    for it, x in enumerate(data['x']):
        oc.write('{:.8e}, '.format(x))
        for state_wf in data['wf']:
            oc.write('{:.8e}, '.format(state_wf[it]))
        oc.write('\n')

    output = {
        'content': oc.getvalue(),
        'filename': 'particle_in_a_box_data.dat'
    }

    return output


def update_save_format(fmt: str):
    '''
    Updates save format of plot.

    Parameters
    ----------
    fmt: str {png, svg, jpeg}
        Image format to use

    Returns
    -------
    Patch
        Patched graph figure
    Patch
        Patched graph config
    '''

    # Figures
    # resetting their layout attr redraws the plot
    # which is necessary because editing the config attr (below) does not...
    fig = Patch()
    fig['layout'] = PIAB_LAYOUT

    # Config
    config = Patch()

    # Update plot format in config dict
    config['toImageButtonOptions']['format'] = fmt

    return fig, config
