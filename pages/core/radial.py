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

FUNC_TO_LABEL = {
    'rdf': 'r<sup>2</sup>R(r)<sup>2</sup>',
    'rwf': 'R(r)'
}

UNIT_TO_LABEL = {
    'bohr': 'a<sub>0</sub>',
    'angstrom': 'Å'
}

UNIT_VALUE = {
    'bohr': 1.,
    'angstrom': 0.529177
}

LABEL_TO_L = {
    's': 0,
    'p': 1,
    'd': 2,
    'f': 3,
}

RAD_TO_LATEX = {
    '1s': r'$$R_{1,0}(\rho) = 2 Z^{\frac{3}{2}} e^{-\rho /2}$$', # noqa
    '2s': r'$$R_{2,0}(\rho) = \frac{Z^{\frac{3}{2}}}{2 \sqrt{2}}e^{-\rho /2} (2-\rho)$$', # noqa
    '2p': r'$$R_{2,1}(\rho) = \frac{Z^{\frac{3}{2}}}{2 \sqrt{6}}e^{-\rho /2} \rho$$', # noqa
    '3s': r'$$R_{3,0}(\rho) = \frac{Z^{\frac{3}{2}}}{9 \sqrt{3}} e^{-\rho /2} (\rho ^2-6 \rho +6)$$', # noqa
    '3p': r'$$R_{3,1}(\rho) = \frac{Z^{\frac{3}{2}}}{9 \sqrt{6}}e^{-\rho /2} (4-\rho ) \rho$$', # noqa
    '3d': r'$$R_{3,2}(\rho) = \frac{Z^{\frac{3}{2}}}{9 \sqrt{30}}e^{-\rho /2} \rho ^2$$', # noqa
    '4s': r'$$R_{4,0}(\rho) = \frac{Z^{\frac{3}{2}}}{96} e^{-\rho /2} (-\rho ^3+12 \rho ^2-36 \rho +24)$$', # noqa
    '4p': r'$$R_{4,1}(\rho) = \frac{Z^{\frac{3}{2}}}{32 \sqrt{15}}e^{-\rho /2} \rho  (\rho ^2-10 \rho +20)$$', # noqa
    '4d': r'$$R_{4,2}(\rho) = \frac{Z^{\frac{3}{2}}}{96 \sqrt{5}}e^{-\rho /2} (6-\rho ) \rho ^2$$', # noqa
    '4f': r'$$R_{4,3}(\rho) = \frac{Z^{\frac{3}{2}}}{96 \sqrt{35}} e^{-\rho /2} \rho ^3$$', # noqa
    '5s': r'$$R_{5,0}(\rho) = \frac{Z^{\frac{3}{2}}}{300 \sqrt{5}} e^{-\rho /2} (\rho ^4-20 \rho ^3+120 \rho ^2-240 \rho +120)$$', # noqa
    '5p': r'$$R_{5,1}(\rho) = \frac{Z^{\frac{3}{2}}}{150 \sqrt{30}}e^{-\rho /2} \rho  (-\rho ^3+18 \rho ^2-90 \rho +120)$$', # noqa
    '5d': r'$$R_{5,2}(\rho) = \frac{Z^{\frac{3}{2}}}{150 \sqrt{70}}e^{-\rho /2} \rho ^2 (\rho ^2-14 \rho +42)$$', # noqa
    '5f': r'$$R_{5,3}(\rho) = \frac{Z^{\frac{3}{2}}}{300 \sqrt{70}}e^{-\rho /2} (8-\rho ) \rho ^3$$', # noqa
    '6s': r'$$R_{6,0}(\rho) = \frac{Z^{\frac{3}{2}}}{2160 \sqrt{6}} e^{-\rho /2} (-\rho ^5+30 \rho ^4-300 \rho ^3+1200 \rho ^2-1800 \rho +720)$$', # noqa
    '6p': r'$$R_{6,1}(\rho) = \frac{Z^{\frac{3}{2}}}{432 \sqrt{210}} e^{-\rho /2} \rho (\rho ^4-28 \rho ^3+252 \rho ^2-840 \rho +840)$$', # noqa
    '6d': r'$$R_{6,2}(\rho) = \frac{Z^{\frac{3}{2}}}{864 \sqrt{105}}e^{-\rho /2} \rho ^2 (-\rho ^3+24 \rho ^2-168 \rho +336)$$', # noqa
    '6f': r'$$R_{6,3}(\rho) = \frac{Z^{\frac{3}{2}}}{2592 \sqrt{35}}e^{-\rho /2} \rho ^3 (\rho ^2-18 \rho +72)$$', # noqa
    '7s': r'$$R_{7,0}(\rho) = \frac{Z^{\frac{3}{2}}}{17640 \sqrt{7}} e^{-\rho /2} (\rho ^6-42 \rho ^5+630 \rho ^4-4200 \rho ^3+12600 \rho ^2-15120 \rho +5040)$$', # noqa
    '7p': r'$$R_{7,1}(\rho) = \frac{Z^{\frac{3}{2}}}{11760 \sqrt{21}}e^{-\rho /2} \rho  (-\rho ^5+40 \rho ^4-560 \rho ^3+3360 \rho ^2-8400 \rho +6720)$$', # noqa
    '7d': r'$$R_{7,2}(\rho) = \frac{Z^{\frac{3}{2}}}{7056 \sqrt{105}}e^{-\rho /2} \rho ^2 (\rho ^4-36 \rho ^3+432 \rho ^2-2016 \rho +3024)$$', # noqa
    '7f': r'$$R_{7,3}(\rho) = \frac{Z^{\frac{3}{2}}}{17640 \sqrt{42}}e^{-\rho /2} \rho ^3 (-\rho ^3+30 \rho ^2-270 \rho +720)$$', # noqa
}

RADIAL_LAYOUT = copy.deepcopy(com.BASIC_LAYOUT)
RADIAL_LAYOUT.xaxis.title = {
    'text': 'r (a<sub>0</sup>)',
    'font': {
        'family': 'Arial',
        'size': 18,
        'color': 'black'
    }
}
RADIAL_LAYOUT.yaxis.title = {
    'text': 'r<sup>2</sup>R(r)<sup>2</sup>',
    'font': {
        'family': 'Arial',
        'size': 18,
        'color': 'black'
    }
}
RADIAL_LAYOUT.yaxis.zeroline = True
RADIAL_LAYOUT.yaxis.tickformat = '.2f'
RADIAL_LAYOUT.xaxis.autorange = False

RADIAL_CONFIG = copy.deepcopy(com.BASIC_CONFIG)


def s_2d(n: int, r: ArrayLike, wf_type: str, z: float) -> NDArray:
    '''
    Calculates s orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r in bohr radii
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_s(n, r, z)**2
    if 'rwf' in wf_type:
        return radial_s(n, r, z)


def p_2d(n: int, r: ArrayLike, wf_type: str, z: float) -> NDArray:
    '''
    Calculates p orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r in bohr radii
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_p(n, r, z)**2
    elif 'rwf' in wf_type:
        return radial_p(n, r, z)


def d_2d(n: int, r: ArrayLike, wf_type: str, z: float) -> NDArray:
    '''
    Calculates d orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r in bohr radii
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_d(n, r, z)**2
    if 'rwf' in wf_type:
        return radial_d(n, r, z)


def f_2d(n: int, r: ArrayLike, wf_type: str, z: float) -> NDArray:
    '''
    Calculates f orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n: int
        prinipal quantum number of orbital
    r: array_like
        values of distance r in bohr radii
    wf_type: str {'RDF', 'RWF'}
        type of wavefunction to calculate
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        y values corresponding to radial wavefunction or radial
        distribution function
    '''

    if 'rdf' in wf_type:
        return r**2. * radial_f(n, r, z)**2
    if 'rwf' in wf_type:
        return radial_f(n, r, z)


def radial_s(n: int, r: ArrayLike, z: float) -> NDArray:
    '''
    Calculates radial Wavefunction of s orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    r: array_like
        Distance in bohr radii where r^2 = x^2+y^2+z^2
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    rho = 2 * z * r / n

    if n == 1:
        rad = z**1.5 * 2. * np.exp(-rho / 2.)
    if n == 2:
        rad = z**1.5 / (2. * np.sqrt(2.)) * (2. - rho) * np.exp(-rho / 2.)
    if n == 3:
        rad = z**1.5 / (9. * np.sqrt(3.)) * (6. - 6. * rho + rho**2.) * np.exp(-rho / 2.) # noqa
    if n == 4:
        rad = z**1.5 / 96. * (24.-36. * rho + 12. * rho**2. - rho**3.) * np.exp(-rho / 2.) # noqa
    if n == 5:
        rad = z**1.5 / (300. * np.sqrt(5.))*(120.-240. * rho + 120. * rho**2.-20. * rho**3. + rho**4.) * np.exp(-rho / 2.) # noqa
    if n == 6:
        rad = z**1.5 / (2160. * np.sqrt(6.))*(720.-1800. * rho + 1200. * rho**2.-300. * rho**3.+30. * rho**4.-rho**5.) * np.exp(-rho / 2.) # noqa
    if n == 7:
        rad = z**1.5 / (17640. * np.sqrt(7))*(5040. - 15120. * rho + 12600. * rho**2. - 4200. * rho**3. + 630. * rho**4. -42* rho**5. + rho**6) * np.exp(-rho / 2.) # noqa

    return rad


def avg_r(n: int, l: int, z: float) -> float:
    '''
    Computes expectation value of r for a given radial wavefunction

    Parameters
    ----------
    n: int
        n quantum number
    l: int
        l quantum number
    z: float
        Z or Z_eff

    Returns
    -------
    float
        <r> for given wavefunction in units of bohr radii
    '''
    return n ** 2 / z * (1 + 0.5 * (1 - (l**2 + l) / (n**2)))


def radial_p(n: int, r: ArrayLike, z: float) -> NDArray:
    '''
    Calculates radial Wavefunction of p orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    r: array_like
        Distance in bohr radii where r^2 = x^2+y^2+z^2
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    rho = 2 * z * r / n

    if n == 2:
        rad = z**1.5 / (2. * np.sqrt(6.)) * rho * np.exp(-rho / 2.)
    elif n == 3:
        rad = z**1.5 / (9. * np.sqrt(6.)) * rho * (4. - rho) * np.exp(-rho / 2.) # noqa
    elif n == 4:
        rad = z**1.5 / (32. * np.sqrt(15.)) * rho * (20.-10. * rho + rho**2.) * np.exp(-rho / 2.) # noqa
    elif n == 5:
        rad = z**1.5 / (150. * np.sqrt(30.)) * rho * (120.-90. * rho + 18. * rho**2. - rho**3.) * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = z**1.5 / (432. * np.sqrt(210.)) * rho * (840.-840. * rho + 252. * rho**2.-28. * rho**3. + rho**4.) * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = z**1.5 / (11760. * np.sqrt(21.)) * rho * (6720. - 8400. * rho + 3360. * rho**2.-560. * rho**3.+40 * rho**4. - rho**5) * np.exp(-rho / 2.) # noqa
    return rad


def radial_d(n: int, r: ArrayLike, z: float) -> NDArray:
    '''
    Calculates radial Wavefunction of d orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    r: array_like
        Distance in bohr radii where r^2 = x^2+y^2+z^2
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    rho = 2 * z * r / n

    if n == 3:
        rad = z**1.5 / (9. * np.sqrt(30.)) * rho**2. * np.exp(-rho / 2.)
    elif n == 4:
        rad = z**1.5 / (96. * np.sqrt(5.))*(6.-rho) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 5:
        rad = z**1.5 / (150. * np.sqrt(70.))*(42.-14. * rho + rho**2) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = z**1.5 / (864. * np.sqrt(105.))*(336.-168. * rho + 24. * rho**2. - rho**3.) * rho**2. * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = z**1.5 / (7056. * np.sqrt(105.))*(3024. - 2016. * rho + 432. * rho**2. -36* rho**3. + rho**4) * rho**2. * np.exp(-rho / 2.) # noqa

    return rad


def radial_f(n: int, r: ArrayLike, z: float) -> NDArray:
    '''
    Calculates radial wavefunction of f orbital
    for the specified principal quantum number

    Parameters
    ----------
    n: int
        principal quantum number
    r: array_like
        Distance in bohr radii where r^2 = x^2+y^2+z^2
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        radial wavefunction as a function of rho
    '''

    rho = 2 * z * r / n

    if n == 4:
        rad = z**1.5 / (96. * np.sqrt(35.)) * rho**3. * np.exp(-rho / 2.)
    elif n == 5:
        rad = z**1.5 / (300. * np.sqrt(70.)) * (8. - rho) * rho**3. * np.exp(-rho / 2.) # noqa
    elif n == 6:
        rad = z**1.5 / (2592. * np.sqrt(35.)) * (rho**2.-18. * rho + 72.) * rho**3. * np.exp(-rho / 2.) # noqa
    elif n == 7:
        rad = z**1.5 / (17640. * np.sqrt(42.)) * (-rho**3 + 30 * rho**2. - 270. * rho + 720.) * rho**3. * np.exp(-rho / 2.) # noqa

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
            max=150,
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

        self.x_unit_select = dbc.Select(
            id=str(uuid.uuid1()),
            options=[
                {'value': 'bohr', 'label': 'Bohr Radii'},
                {'value': 'angstrom', 'label': 'Angstrom'}
            ],
            value='angstrom',
            style={'textAlign': 'center'}
        )

        self.x_unit_ig = self.make_input_group(
            [
                dbc.InputGroupText('Distance unit'),
                self.x_unit_select
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

        self.z_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=1,
            value=1,
            min=1,
            type='number',
            style={'textAlign': 'center'}
        )

        self.z_ig = self.make_input_group(
            [
                dbc.InputGroupText('Z'),
                self.z_input
            ],
            justify=True
        )

        # Legend toggle checkbox
        self.legend_toggle = dbc.Checkbox(
            value=True,
            id=str(uuid.uuid1()),
        )
        self.legend_ig = self.make_input_group(
            [
                dbc.InputGroupText('Legend'),
                dbc.InputGroupText(
                    self.legend_toggle
                )
            ],
            justify=True
        )

        # Average r toggle checkbox
        self.avg_distance_toggle = dbc.Checkbox(
            value=True,
            id=str(uuid.uuid1())
        )
        self.avg_distance_ig = self.make_input_group(
            [
                dbc.InputGroupText('⟨r⟩'),
                dbc.InputGroupText(
                    self.avg_distance_toggle
                )
            ],
            justify=True
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

        self.font_size_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=18,
            value=18,
            min=10,
            type='number',
            style={'textAlign': 'center'}
        )

        self.font_size_ig = self.make_input_group(
            [
                dbc.InputGroupText('Font Size'),
                self.font_size_input
            ]
        )

        self.linewidth_input = dbc.Input(
            id=str(uuid.uuid1()),
            placeholder=2.,
            type='number',
            min=1,
            max=15,
            value=5,
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
            ],
            justify=True
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

    def make_input_group(self, elements, justify=False):

        group = dbc.InputGroup(
            id=str(uuid.uuid1()),
            children=elements,
            class_name='mb-3'
        )

        if justify:
            group.style = {'justify-content': 'center'}

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
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        self.func_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    )
                ],
                justify='center'
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        self.upper_x_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        self.x_unit_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    )
                ],
                justify='center'
            ),
            dbc.Row(
                [
                    dbc.Col(
                        self.colour_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        self.z_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=3
                    ),
                    dbc.Col(
                        [
                            self.avg_distance_ig,
                            dbc.Tooltip(
                                target=self.avg_distance_ig.id,
                                children='Add line for average electron distance' # noqa
                            )
                        ],
                        class_name='mb-3 text-center mwmob',
                        sm=3,
                    )
                ],
                justify='center'
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            self.font_size_ig,
                        ],
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=4
                    ),
                    dbc.Col(
                        [
                            self.linewidth_ig,
                        ],
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=4
                    ),
                    dbc.Col(
                        self.legend_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=4
                    ),
                ],
                justify='center'
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
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        [
                            self.download_button,
                            self.download_trigger
                        ],
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Tooltip(
                        children='Use the camera button in the top right of \
                                the plot to save an image',
                        target=self.image_format_ig.id,
                        style={
                            'textAlign': 'center',
                        },
                    )
                ],
                justify='center'
            )
        ]

        self.div.children = contents
        return


def assemble_callbacks(plot_div: com.PlotDiv, options: OptionsDiv) -> None:
    '''
    Creates callbacks between experimental and fitted AC plots and elements in
    AC options tab

    Parameters
    ----------
    plot_div: Orb2dPlot
        Experimental plot tab object
    options: OptionsDiv
        Options div object

    Returns
    -------
    None
    '''

    # Callback for download 2d data button
    states = [
        State(options.func_select, 'value'),
        State(options.orb_select, 'value'),
        State(options.x_unit_select, 'value'),
        State(options.upper_x_input, 'value'),
        State(options.z_input, 'value')
    ]
    callback(
        Output(options.download_trigger, 'data'),
        Input(options.download_button, 'n_clicks'),
        states,
        prevent_initial_call=True
    )(download_data)

    # Callback for plotting data
    inputs = [
        Input(options.func_select, 'value'),
        Input(options.orb_select, 'value'),
        Input(options.upper_x_input, 'value'),
        Input(options.x_unit_select, 'value'),
        Input(options.colour_select, 'value'),
        Input(options.legend_toggle, 'value'),
        Input(options.avg_distance_toggle, 'value'),
        Input(options.font_size_input, 'value'),
        Input(options.linewidth_input, 'value'),
        Input(options.z_input, 'value')
    ]
    callback(
        [
            Output(plot_div.plot, 'figure', allow_duplicate=True),
            Output(plot_div.plot, 'config', allow_duplicate=True)
        ],
        inputs,
        prevent_initial_call='initial_duplicate'
    )(update_plot)

    # Update plot save format
    callback(
        [
            Output(plot_div.plot, 'figure', allow_duplicate=True),
            Output(plot_div.plot, 'config', allow_duplicate=True),
        ],
        [
            Input(options.image_format_select, 'value')
        ],
        [
            State(options.func_select, 'value'),
            State(options.x_unit_select, 'value'),
            State(options.upper_x_input, 'value')
        ],
        prevent_initial_call=True
    )(update_save_format)

    return


def compute_radials(func: str, orbs: list[str], r: ArrayLike,
                    z: float) -> NDArray:
    '''
    Computes radial wavefunction or radial distribution function

    Parameters
    ----------
    func : str {'Radial Wavefunction', 'Radial Distribution Function'}
        Name of function
    orbs: list[str]
        Orbitals to include
    r: array_like
        r values in Bohr radii
    z: float
        Z or Z_eff

    Returns
    -------
    ndarray of floats
        orbital functions at each r, with same order as `orbs`
    '''

    if isinstance(orbs, str):
        orbs = [orbs]

    # Dictionary of orbital functions
    orb_funcs = {
        's': s_2d,
        'p': p_2d,
        'd': d_2d,
        'f': f_2d,
    }

    full_data = [
        orb_funcs[orb[1]](int(orb[0]), r, func, z)
        for orb in orbs
    ]

    return full_data


def download_data(_nc: int, func: str, orbs: list[str], unit: str,
                  x_max: float, z: float) -> dict:
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
    unit: str {'Angstrom', 'Bohr Radii'}
        Distance unit
    x_max: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)
    z: float
        Z or Z_eff

    Returns
    -------
    dict
        Output dictionary used by dcc.Download
    '''

    if None in [unit, func, orbs, z]:
        return no_update

    if not len(orbs):
        return no_update

    # r in Bohr radii
    r = np.linspace(0, x_max / UNIT_VALUE[unit], 5000)

    # R(r) or RDF where r has units of Bohr radii
    radial = np.array(compute_radials(func, orbs, r, z))

    func_to_name = {
        'rdf': 'Radial Distribution Function',
        'rwf': 'Radial Wavefunction'
    }

    # Comments
    comment = '{}\n'.format(func_to_name[func])
    comment += 'Data generated using waveplot.com\n'
    comment += 'an app by Jon Kragskow \n '

    # Column headers
    header = f'r ({unit.capitalize()}),'
    for orb in orbs:
        header += '{}, '.format(orb)

    # Create iostring and save data to it
    data_str = io.StringIO()
    data = np.vstack([r * UNIT_VALUE[unit], radial]).T
    np.savetxt(data_str, data, comments=comment, header=header, delimiter=',')

    # Create output dictionary for dcc.Download
    output = {
        'content': data_str.getvalue(),
        'filename': 'radial_wavefunction_data.csv'
    }

    return output


def update_plot(func: str, orbs: list[str], x_max: float, unit: str,
                colour_scheme: str, legend: bool,
                avg_distance: bool, font_size: float,
                lw: float, z: float) -> tuple[Patch, Patch]:
    '''
    Plots Radial wavefunction/distribution function data

    Parameters
    ----------
    func : str {'Radial Wavefunction', 'Radial Distribution Function'}
        Name of function
    orbs: list[str]
        Orbitals to include
    x_max: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)
    unit: str {'Angstrom', 'Bohr Radii'}
        Distance unit
    colour_scheme: str ['tol', 'wong', 'standard']
        Colour scheme to use for plots
    legend: bool
        If True, displays legend
    avg_distance: bool
        If True, plots line for average radial distance
    font_size: float
        Font size for axis and tick labels
    lw: float
        Linewidth of traces
    z: float
        Z or Z_eff

    Returns
    -------
    Patch
        Patched graph figure
    Patch
        Patched graph config
    '''

    fig = Patch()

    if None in [x_max, unit, func, orbs, font_size, z]:
        return no_update

    if not len(orbs):
        return no_update

    # Create colour list in correct order
    # i.e. selected colour is first
    if colour_scheme == 'tol':
        colours = ut.tol_cols + ut.wong_cols + ut.def_cols
    elif colour_scheme == 'wong':
        colours = ut.wong_cols + ut.def_cols + ut.tol_cols
    else:
        colours = ut.def_cols + ut.tol_cols + ut.wong_cols

    # r in Bohr radii
    r = np.linspace(0, x_max / UNIT_VALUE[unit], 5000)

    # R(r) or RDF where r has units of Bohr radii
    radial = compute_radials(func, orbs, r, z)

    # Average distance in Bohr radii
    average_r = [
        avg_r(int(orb[0]), LABEL_TO_L[orb[1]], z)
        for orb in orbs
    ]

    # Plot R(r) or RDF
    traces = [
        go.Scatter(
            x=r * UNIT_VALUE[unit],
            y=rad,
            line={
                'width': lw
            },
            name=orb,
            legendgroup=orb,
            hoverinfo='skip',
            marker={'color': col}
        )
        for rad, orb, col in zip(radial, orbs, colours)
    ]

    if avg_distance:
        avg_traces = [
            go.Scatter(
                x=[avgr * UNIT_VALUE[unit]] * 100,
                y=np.linspace(0, np.max(radial), 100),
                mode='lines',
                legendgroup=orb,
                showlegend=False,
                hovertext=f'{orb} ⟨r⟩ = {avgr * UNIT_VALUE[unit]:.2f} {UNIT_TO_LABEL[unit]}', # noqa
                hoverinfo='text',
                name=orb,
                line={'color': col, 'dash': 'dash', 'width': lw * 0.5}
            )
            for avgr, col, orb in zip(average_r, colours, orbs)
        ]

        traces += avg_traces

    fig['data'] = traces

    # Update x axis with correct unit
    fig['layout']['xaxis']['title']['text'] = f'r ({UNIT_TO_LABEL[unit]})'
    fig['layout']['xaxis']['range'] = [0, x_max]

    # Update y axis with correct label
    fig['layout']['yaxis']['title']['text'] = FUNC_TO_LABEL[func]

    fig['layout']['showlegend'] = legend

    fig.layout.legend.font.size = font_size * 0.75
    fig.layout.xaxis.tickfont.size = font_size
    fig.layout.yaxis.tickfont.size = font_size
    fig.layout.yaxis.title.font.size = font_size
    fig.layout.xaxis.title.font.size = font_size

    func_to_fname = {
        'rdf': 'radial_distribution_function',
        'rwf': 'radial_wave_function'
    }

    config = Patch()

    config['toImageButtonOptions']['filename'] = func_to_fname[func]

    return fig, config


def update_save_format(fmt: str, func: str, unit: str, x_max: int):
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
    x_max: float
        Upper x value in either Angstrom or Bohr radii (depends on `unit`)

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

    # Update x axis with correct unit
    fig['layout']['xaxis']['title']['text'] = f'r ({UNIT_TO_LABEL[unit]})'

    # Update y axis with correct label
    fig['layout']['yaxis']['title']['text'] = FUNC_TO_LABEL[func]
    fig['layout']['xaxis']['range'] = [0, x_max]

    # Config
    config = Patch()

    # Update plot format in config dict
    config['toImageButtonOptions']['format'] = fmt

    return fig, config
