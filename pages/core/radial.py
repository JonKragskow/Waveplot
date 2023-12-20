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
from numpy.typing import NDArray, ArrayLike
from dash import dcc, html, Input, Output, State, callback, no_update, Patch
import dash_bootstrap_components as dbc
import io
import plotly.graph_objects as go
import copy
import uuid

from . import common as com
from . import utils as ut


RADIAL_LAYOUT = copy.deepcopy(com.BASIC_LAYOUT)
RADIAL_LAYOUT.xaxis.title = {
    'text': 'r (a<sub>0</sup>)',
    'font': {
        'family': 'Arial',
        'size': 14,
        'color': 'black'
    }
}
RADIAL_LAYOUT.yaxis.title = {
    'text': '4πr<sup>2</sup>R(r)<sup>2</sup>',
    'font': {
        'family': 'Arial',
        'size': 14,
        'color': 'black'
    }
}
RADIAL_LAYOUT.yaxis.zeroline = True
RADIAL_LAYOUT.yaxis.tickformat = '.2f'

RADIAL_CONFIG = copy.deepcopy(com.BASIC_CONFIG)


def s_2d(n: int, r: ArrayLike, wf_type: str) -> NDArray:
    '''
    Calculates s orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_s(n, 2. * r / n)**2
    if 'rwf' in wf_type:
        return radial_s(n, 2. * r / n)


def p_2d(n: int, r: ArrayLike, wf_type: str) -> NDArray:
    '''
    Calculates p orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_p(n, 2. * r / n)**2
    elif 'rwf' in wf_type:
        return radial_p(n, 2. * r / n)


def d_2d(n: int, r: ArrayLike, wf_type: str) -> NDArray:
    '''
    Calculates d orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_d(n, 2. * r / n)**2
    if 'rwf' in wf_type:
        return radial_d(n, 2. * r / n)


def f_2d(n: int, r: ArrayLike, wf_type: str) -> NDArray:
    '''
    Calculates f orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_f(n, 2. * r / n)**2
    if 'rwf' in wf_type:
        return radial_f(n, 2. * r / n)


def radial_s(n: int, rho: ArrayLike):
    '''
    Calculates radial Wavefunction of s orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    rho: array_like
        values of rho = 2. * r / n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    if n == 1:
        rad = 2. * np.exp(-rho / 2.)
    if n == 2:
        rad = 1. / (2. * np.sqrt(2.)) * (2. - rho) * np.exp(-rho / 2.)
    if n == 3:
        rad = 1. / (9. * np.sqrt(3.)) * (6. - 6. * rho + rho**2.) * np.exp(-rho / 2.) # noqa
    if n == 4:
        rad = (1. / 96.) * (24.-36. * rho + 12. * rho**2. - rho**3.) * np.exp(-rho / 2.) # noqa
    if n == 5:
        rad = (1. / (300. * np.sqrt(5.)))*(120.-240. * rho + 120. * rho**2.-20. * rho**3. + rho**4.) * np.exp(-rho / 2.) # noqa
    if n == 6:
        rad = (1. / (2160. * np.sqrt(6.)))*(720.-1800. * rho + 1200. * rho**2.-300. * rho**3.+30. * rho**4.-rho**5.) * np.exp(-rho / 2.) # noqa
    if n == 7:
        rad = (1. / (17640. * np.sqrt(7)))*(5040. - 15120. * rho + 12600. * rho**2. - 4200. * rho**3. + 630. * rho**4. -42* rho**5. + rho**6) * np.exp(-rho / 2.) # noqa

    return rad


def radial_p(n: int, rho: ArrayLike) -> NDArray:
    '''
    Calculates radial Wavefunction of p orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    rho: array_like
        values of rho = 2. * r / n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    if n == 2:
        rad = 1. / (2. * np.sqrt(6.)) * rho * np.exp(-rho / 2.)
    elif n == 3:
        rad = 1. / (9. * np.sqrt(6.)) * rho * (4. - rho) * np.exp(-rho / 2.)
    elif n == 4:
        rad = 1. / (32. * np.sqrt(15.)) * rho * (20.-10. * rho + rho**2.) * np.exp(-rho / 2.) # noqa
    elif n == 5:
        rad = 1. / (150. * np.sqrt(30.)) * rho * (120.-90. * rho + 18. * rho**2. - rho**3.) * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = 1. / (432. * np.sqrt(210.)) * rho * (840.-840. * rho + 252. * rho**2.-28. * rho**3. + rho**4.) * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = 1. / (11760. * np.sqrt(21.)) * rho * (6720. - 8400. * rho + 3360. * rho**2.-560. * rho**3.+40 * rho**4. - rho**5) * np.exp(-rho / 2.) # noqa
    return rad


def radial_d(n: int, rho: ArrayLike) -> NDArray:
    '''
    Calculates radial Wavefunction of d orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    rho: array_like
        values of rho = 2. * r / n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    if n == 3:
        rad = 1. / (9. * np.sqrt(30.)) * rho**2. * np.exp(-rho / 2.)
    elif n == 4:
        rad = 1. / (96. * np.sqrt(5.))*(6.-rho) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 5:
        rad = 1. / (150. * np.sqrt(70.))*(42.-14. * rho + rho**2) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = 1. / (864. * np.sqrt(105.))*(336.-168. * rho + 24. * rho**2. - rho**3.) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = 1. / (7056. * np.sqrt(105.))*(3024. - 2016. * rho + 432. * rho**2. -36* rho**3. + rho**4) * rho**2. * np.exp(-rho / 2.) # noqa

    return rad


def radial_f(n: int, rho: ArrayLike) -> NDArray:
    '''
    Calculates radial wavefunction of f orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    rho: array_like
        values of rho = 2. * r / n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    if n == 4:
        rad = 1. / (96. * np.sqrt(35.)) * rho**3. * np.exp(-rho / 2.)
    elif n == 5:
        rad = 1. / (300. * np.sqrt(70.)) * (8. - rho) * rho**3. * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = 1. / (2592. * np.sqrt(35.)) * (rho**2.-18. * rho + 72.) * rho**3. * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = 1. / (17640. * np.sqrt(42.)) * (-rho**3 + 30 * rho**2. - 270. * rho + 720.) * rho**3. * np.exp(-rho / 2.) # noqa

    return rad


class OptionsDiv(com.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.orb_select = dcc.Dropdown(
            id=str(uuid.uuid1()),
            style={
                'textAlign': 'left'
            },
            options=[
                {'label': '1s', 'value': '1s'},
                {'label': '2s', 'value': '2s'},
                {'label': '3s', 'value': '3s'},
                {'label': '4s', 'value': '4s'},
                {'label': '5s', 'value': '5s'},
                {'label': '6s', 'value': '6s'},
                {'label': '7s', 'value': '7s'},
                {'label': '2p', 'value': '2p'},
                {'label': '3p', 'value': '3p'},
                {'label': '4p', 'value': '4p'},
                {'label': '5p', 'value': '5p'},
                {'label': '6p', 'value': '6p'},
                {'label': '7p', 'value': '7p'},
                {'label': '3d', 'value': '3d'},
                {'label': '4d', 'value': '4d'},
                {'label': '5d', 'value': '5d'},
                {'label': '6d', 'value': '6d'},
                {'label': '7d', 'value': '7d'},
                {'label': '4f', 'value': '4f'},
                {'label': '5f', 'value': '5f'},
                {'label': '6f', 'value': '6f'},
                {'label': '7f', 'value': '7f'},
            ],
            value=['1s', '2p', '3d', '4f'],
            multi=True,
            placeholder='Select an Orbital'
        )

        self.func_select = dbc.Select(
            id=str(uuid.uuid1()),
            style={
                'textAlign': 'center',
                'display': 'block'
            },
            options=[
                {
                    'label': 'Radial Distribution Function',
                    'value': 'rdf'
                },
                {
                    'label': 'Radial Wave Function',
                    'value': 'rwf'
                }
            ],
            value='rdf',
        )

        self.func_ig = self.make_input_group(
            [
                dbc.InputGroupText('Function'),
                self.func_select
            ]
        )

        self.upper_x_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=0,
            type='number',
            min=0,
            max=100,
            value=20,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.upper_x_ig = self.make_input_group(
            [
                dbc.InputGroupText('Upper x limit'),
                self.upper_x_input
            ]
        )

        self.distance_select = dbc.Select(
            id=str(uuid.uuid1()),
            options=[
                {'value': 'bohr', 'label': 'Bohr Radii'},
                {'value': 'angstrom', 'label': 'Angstrom'}
            ],
            value='angstrom',
            style={'textAlign': 'center'}
        )

        self.distance_ig = self.make_input_group(
            [
                dbc.InputGroupText('Distance unit'),
                self.distance_select
            ]
        )

        self.colour_select = dbc.Select(
            id=str(uuid.uuid1()),
            options=[
                {
                    'label': 'Standard',
                    'value': 'normal'
                },
                {
                    'label': 'Tol',
                    'value': 'tol'
                },
                {
                    'label': 'Wong',
                    'value': 'wong'
                }
            ],
            value='normal',
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle',
                'alignItems': 'auto',
                'display': 'inline'
            }
        )

        self.colour_ig = self.make_input_group(
            [
                dbc.InputGroupText('Colour Palette'),
                self.colour_select
            ]
        )

        # Legend toggle checkbox
        self.legend_toggle = dbc.Checkbox(
            value=True,
            id=str(uuid.uuid1()),
        )
        self.legend_ig = dbc.InputGroup(
            children=[
                dbc.InputGroupText('Legend'),
                dbc.InputGroupText(
                    self.legend_toggle
                )
            ]
        )

        self.download_button = dbc.Button(
            'Download Data',
            id=str(uuid.uuid1()),
            style={
                'boxShadow': 'none',
                'width': '100%'
            }
        )
        self.download_trigger = dcc.Download(
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
                        self.orb_select,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.func_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        self.upper_x_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.distance_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        self.colour_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.legend_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row([
                dbc.Col([
                    html.H4(
                        style={
                            'textAlign': 'center',
                            'margin-bottom': '5%',
                            'margin-top': '5%'
                        },
                        children='Output'
                    )
                ])
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        self.image_format_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        [
                            self.download_button,
                            self.download_trigger
                        ],
                        class_name='mb-3'
                    ),
                    dbc.Tooltip(
                        children='Use the camera button in the top right of \
                                the plot to save an image',
                        target=self.image_format_ig.id,
                        style={
                            'textAlign': 'center',
                        },
                    )
                ]
            )
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: com.PlotDiv, options_div: OptionsDiv) -> None:
    '''
    Creates callbacks between experimental and fitted AC plots and elements in
    AC options tab

    Parameters
    ----------
    plot_div: Orb2dPlot
        Experimental plot tab object
    options_div: OptionsDiv
        Options div object

    Returns
    -------
    None
    '''

    # Callback for download 2d data button
    states = [
        State(options_div.func_select, 'value'),
        State(options_div.orb_select, 'value'),
        State(options_div.upper_x_input, 'value'),
        State(options_div.distance_select, 'value')
    ]
    callback(
        Output(options_div.download_trigger, 'data'),
        Input(options_div.download_button, 'n_clicks'),
        states,
        prevent_initial_call=True
    )(download_data)

    # Callback for plotting data
    inputs = [
        Input(options_div.func_select, 'value'),
        Input(options_div.orb_select, 'value'),
        Input(options_div.upper_x_input, 'value'),
        Input(options_div.distance_select, 'value'),
        Input(options_div.colour_select, 'value'),
        Input(options_div.legend_toggle, 'value')
    ]
    callback(
        [
            Output(plot_div.plot, 'figure', allow_duplicate=True),
            Output(plot_div.plot, 'config', allow_duplicate=True),
        ],
        inputs,
        prevent_initial_call='initial_duplicate'
    )(plot_data)

    # Update plot save format
    callback(
        [
            Output(plot_div.plot, 'figure', allow_duplicate=True),
            Output(plot_div.plot, 'config', allow_duplicate=True),
        ],
        [
            Input(options_div.image_format_select, 'value')
        ],
        [
            State(options_div.func_select, 'value'),
            State(options_div.distance_select, 'value')
        ],
        prevent_initial_call=True
    )(update_save_format)

    return


def compute_radials(func: str, orbs: list[str], low_x: float, up_x: float,
                    unit: str) -> NDArray:
    '''
    Computes radial wavefunction or radial distribution function

    Parameters
    ----------
    func : str {'Radial Wavefunction', 'Radial Distribution Function'}
        Name of function
    orbs: list[str]
        Orbitals to include
    low_x: float
        Lower x value in either Angstrom or Bohr radii (depends on `unit`)
    up_x: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)
    unit: str {'Angstrom', 'Bohr Radii'}
        Distance unit

    Returns
    -------
    ndarray of floats
        First column is x, remaining columns are orbital data with\n
        same order as `orbs`
    '''

    unit_conv = {
        'bohr': 1.,
        'angstrom': 0.529177
    }

    low_x /= unit_conv[unit]
    up_x /= unit_conv[unit]

    x = np.linspace(low_x, up_x, 5000)

    # Dictionary of orbital functions
    orb_funcs = {
        's': s_2d,
        'p': p_2d,
        'd': d_2d,
        'f': f_2d,
    }

    full_data = np.zeros([x.size, len(orbs) + 1])
    full_data[:, 0] = x * unit_conv[unit]

    for it, orb in enumerate(orbs):
        full_data[:, it + 1] = orb_funcs[orb[1]](int(orb[0]), x, func)

    return full_data


def download_data(_nc: int, func: str, orbs: list[str],
                  up_x: float, unit: str) -> dict:
    '''
    Creates output file for Radial wavefunction/distribution function

    Parameters
    ----------
    _nc: int
        Number of clicks on download button. Used only to trigger callback
    func : str {'Radial Wavefunction', 'Radial Distribution Function'}
        Name of function
    orbs: list[str]
        Orbitals to include
    up_x: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)
    unit: str {'Angstrom', 'Bohr Radii'}
        Distance unit

    Returns
    -------
    dict
        Output dictionary used by dcc.Download
    '''

    if None in [up_x, unit, func, orbs]:
        return no_update

    if not len(orbs):
        return no_update

    # Recompute the data
    data = compute_radials(func, orbs, 0., up_x, unit)

    func_to_name = {
        'rdf': 'Radial Distribution Function',
        'rwf': 'Radial Wavefunction'
    }

    # Comments
    comment = '{}\n'.format(func_to_name[func])
    comment += 'Data generated using waveplot.com\n'
    comment += 'an app by Jon Kragskow \n '

    # Column headers
    header = f'x ({unit}),'
    for orb in orbs:
        header += '{}, '.format(orb)

    # Create iostring and save data to it
    data_str = io.StringIO()
    np.savetxt(data_str, data, comments=comment, header=header, delimiter=',')

    # Create output dictionary for dcc.Download
    output = {
        'content': data_str.getvalue(),
        'filename': 'waveplot_orbital_data.csv'
    }

    return output


def plot_data(func: str, orbs: list[str], up_x: float,
              unit: str, colour_scheme: str,
              legend: bool) -> tuple[Patch, Patch]:
    '''
    Plots Radial wavefunction/distribution function data

    Parameters
    ----------
    func : str {'Radial Wavefunction', 'Radial Distribution Function'}
        Name of function
    orbs: list[str]
        Orbitals to include
    up_x: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)
    unit: str {'Angstrom', 'Bohr Radii'}
        Distance unit
    colour_scheme: str ['tol', 'wong', 'standard']
        Colour scheme to use for plots
    legend: bool
        Display legend (on=True, off=False)

    Returns
    -------
    Patch
        Patched graph figure
    Patch
        Patched graph config
    '''

    fig = Patch()

    if None in [up_x, unit, func, orbs]:
        return no_update

    if not len(orbs):
        return no_update

    # Create colour list in correct order
    # i.e. selected colour is first
    if colour_scheme == 'tol':
        cols = ut.tol_cols + ut.wong_cols + ut.def_cols
    elif colour_scheme == 'wong':
        cols = ut.wong_cols + ut.def_cols + ut.tol_cols
    else:
        cols = ut.def_cols + ut.tol_cols + ut.wong_cols

    data = compute_radials(func, orbs, 0., up_x, unit)

    # Create plotly trace
    traces = [
        go.Scatter(
            x=data[:, 0],
            y=data[:, it + 1],
            line={
                'width': 5
            },
            name=orb,
            hoverinfo='none',
            marker={'color': cols[it]}
        )
        for it, orb in enumerate(orbs)
    ]

    fig['data'] = traces

    unit_to_label = {
        'bohr': 'r (a<sub>0</sub>)',
        'angstrom': 'r (Å)'
    }

    func_to_label = {
        'rdf': '4πr<sup>2</sup>R(r)<sup>2</sup>',
        'rwf': 'R(r)'
    }

    # Update x axis with correct unit
    fig['layout']['xaxis']['title']['text'] = unit_to_label[unit]

    # Update y axis with correct label
    fig['layout']['yaxis']['title']['text'] = func_to_label[func]

    fig['layout']['showlegend'] = legend

    func_to_fname = {
        'rdf': 'radial_distribution_function',
        'rwf': 'radial_wave_function'
    }

    config = Patch()

    config['toImageButtonOptions']['filename'] = func_to_fname[func]

    return fig, config


def update_save_format(fmt: str, func: str, unit: str):
    '''
    Updates save format of plot.

    Parameters
    ----------
    fmt: str {png, svg, jpeg}
        Image format to use
    func: str {rdf, rwf}
        Type of function being plotted
    unit: str {'bohr', 'angstrom'}
        x unit used for data

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
    fig['layout'] = RADIAL_LAYOUT

    func_to_label = {
        'rdf': '4πr<sup>2</sup>R(r)<sup>2</sup>',
        'rwf': 'R(r)'
    }

    unit_to_label = {
        'bohr': 'r (a<sub>0</sub>)',
        'angstrom': 'r (Å)'
    }

    # Update x axis with correct unit
    fig['layout']['xaxis']['title']['text'] = unit_to_label[unit]

    # Update y axis with correct label
    fig['layout']['yaxis']['title']['text'] = func_to_label[func]

    # Config
    config = Patch()

    # Update plot format in config dict
    config['toImageButtonOptions']['format'] = fmt

    return fig, config
