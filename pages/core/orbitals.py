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
import numpy as np
from . import common
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc


def radial_s(n, rho):
    """
    Calculates radial Wavefunction of s orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 1:
        rad = 2.*np.exp(-rho/2.)
    if n == 2:
        rad = 1./(2.*np.sqrt(2.))*(2.-rho)*np.exp(-rho/2.)
    if n == 3:
        rad = 1./(9.*np.sqrt(3.))*(6.-6.*rho+rho**2.)*np.exp(-rho/2.)
    if n == 4:
        rad = (1./96.)*(24.-36.*rho+12.*rho**2.-rho**3.)*np.exp(-rho/2.)
    if n == 5:
        rad = (1./(300.*np.sqrt(5.)))*(120.-240.*rho+120.*rho**2.-20.*rho**3.+rho**4.)*np.exp(-rho/2.) # noqa
    if n == 6:
        rad = (1./(2160.*np.sqrt(6.)))*(720.-1800.*rho+1200.*rho**2.-300.*rho**3.+30.*rho**4.-rho**5.)*np.exp(-rho/2.) # noqa
    if n == 7:
        rad = (1./(17640.*np.sqrt(7)))*(5040. - 15120.*rho + 12600.*rho**2. - 4200.*rho**3. + 630.*rho**4. -42* rho**5. + rho**6)*np.exp(-rho/2.) # noqa

    return rad


def radial_p(n, rho):
    """
    Calculates radial Wavefunction of p orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 2:
        rad = 1./(2.*np.sqrt(6.))*rho*np.exp(-rho/2.)
    elif n == 3:
        rad = 1./(9.*np.sqrt(6.))*rho*(4.-rho)*np.exp(-rho/2.)
    elif n == 4:
        rad = 1./(32.*np.sqrt(15.))*rho*(20.-10.*rho+rho**2.)*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(150.*np.sqrt(30.))*rho*(120.-90.*rho+18.*rho**2.-rho**3.)*np.exp(-rho/2.) # noqa
    elif n == 6:
        rad = 1./(432.*np.sqrt(210.))*rho*(840.-840.*rho+252.*rho**2.-28.*rho**3.+rho**4.)*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(11760.*np.sqrt(21.))*rho*(6720. - 8400.*rho+3360.*rho**2.-560.*rho**3.+40*rho**4. - rho**5)*np.exp(-rho/2.) # noqa
    return rad


def radial_d(n, rho):
    """
    Calculates radial Wavefunction of d orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 3:
        rad = 1./(9.*np.sqrt(30.))*rho**2.*np.exp(-rho/2.)
    elif n == 4:
        rad = 1./(96.*np.sqrt(5.))*(6.-rho)*rho**2.*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(150.*np.sqrt(70.))*(42.-14.*rho+rho**2)*rho**2.*np.exp(-rho/2.) # noqa
    elif n == 6:
        rad = 1./(864.*np.sqrt(105.))*(336.-168.*rho+24.*rho**2.-rho**3.)*rho**2.*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(7056.*np.sqrt(105.))*(3024. - 2016.*rho + 432.*rho**2. -36* rho**3. + rho**4)*rho**2.*np.exp(-rho/2.) # noqa

    return rad


def radial_f(n, rho):
    """
    Calculates radial wavefunction of f orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 4:
        rad = 1./(96.*np.sqrt(35.))*rho**3.*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(300.*np.sqrt(70.))*(8.-rho)*rho**3.*np.exp(-rho/2.)
    elif n == 6:
        rad = 1./(2592.*np.sqrt(35.))*(rho**2.-18.*rho+72.)*rho**3.*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(17640.*np.sqrt(42.))*(-rho**3 + 30*rho**2. - 270.*rho + 720.)*rho**3.*np.exp(-rho/2.) # noqa

    return rad


def s_3d(n, cutaway=1.):
    """
    Calculates s orbital wavefunction on a grid

    Parameters
    ----------
        n : int
            prinipal quantum number of orbital
        cutaway : int
            number used to split orbital in half
    Returns
    -------
        x : np.mgrid
            x values
        y : np.mgrid
            y values
        z : np.mgrid
            z values
        wav : np.mgrid
            wavefunction values at x, y, z
        upper : float
            max value of axes
        lower : float
            min value of axes
        ival : float
            isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_s(n, 2*r/n)

    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.5/np.sqrt(np.pi)

    wav = ang*rad

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.0005, step


def p_3d(n, cutaway=1.):
    """
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_p(n, 2*r/n)

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

    return n_points, wav, upper, lower, ival, step


def dz_3d(n, cutaway=1.):
    """
    Calculates dz2 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_d(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 2*z**2-x**2-y**2

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.08, step


def dxy_3d(n, cutaway=1.):
    """
    Calculates dxy orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_d(n, 2*r/n)
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

    return n_points, wav, upper, lower, ival, step


def fz_3d(n, cutaway=1.):
    """
    Calculates fz3 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 4:
        upper = 70.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 100.
        step = 2*upper/75.
        lower = - upper
    elif n == 6:
        upper = 130.
        step = 2*upper/85.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(7/np.pi) * z*(2*z**2-3*x**2-3*y**2)/(r**3)

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.000005, step


def fxyz_3d(n, cutaway=1.):
    """
    Calculates fxyz orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
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

    return n_points, wav, upper, lower, ival, step


def fyz2_3d(n, cutaway=1.):
    """
    Calculates fyz2 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

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

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(35/(2*np.pi)) * (3*x**2-y**2)*y/r**3

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.000005, step


def s_2d(n, r, wf_type):
    """
    Calculates s orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate
    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_s(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_s(n, 2.*r/n)


def p_2d(n, r, wf_type):
    """
    Calculates p orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate
    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_p(n, 2.*r/n)**2
    elif "RWF" in wf_type:
        return radial_p(n, 2.*r/n)


def d_2d(n, r, wf_type):
    """
    Calculates d orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_d(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_d(n, 2.*r/n)


def f_2d(n, r, wf_type):
    """
    Calculates f orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_f(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_f(n, 2.*r/n)


class Orb2dPlot(common.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.plot = dcc.Graph(
            id=self.prefix('2d_plot'),
            className='plot_area',
            mathjax=True
        )

        self.make_div_contents()

    def make_div_contents(self):
        '''
        Assembles div children in rows and columns
        '''

        contents = [
            dbc.Row(
                [
                    dbc.Col(
                        self.plot,
                    )
                ]
            )
        ]

        self.div.children = contents
        return


class OptionsDiv(common.Div):
    def __init__(self, prefix, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.orb_select = dcc.Dropdown(
            id=self.prefix('orb_select'),
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
            placeholder='Orbital...'
        )

        self.func_select = dbc.Select(
            id=self.prefix('function_type'),
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

        self.lower_x_input = dbc.Input(
            id=self.prefix('lower_x_in'),
            placeholder=0,
            type='number',
            min=-10,
            max=100,
            value=0,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.lower_x_ig = self.make_input_group(
            [
                dbc.InputGroupText('Lower x limit'),
                self.lower_x_input
            ]
        )

        self.upper_x_input = dbc.Input(
            id=self.prefix('upper_x_in'),
            placeholder=0,
            type='number',
            min=0,
            max=100,
            value=40,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.upper_x_ig = self.make_input_group(
            [
                dbc.InputGroupText('Lower x limit'),
                self.upper_x_input
            ]
        )

        self.distance_select = dbc.Select(
            id=self.prefix('distance_unit'),
            options=[
                {'value': 'a0', 'label': 'Bohr Radii'},
                {'value': 'Å', 'label': 'Angstrom'}
            ],
            value='a0',
            style={'textAlign': 'center'}
        )

        self.distance_ig = self.make_input_group(
            [
                dbc.InputGroupText('Distance unit'),
                self.distance_select
            ]
        )

        self.colour_select = dbc.Select(
            id=self.prefix('colours_2d'),
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

        self.output_height_input = dbc.Input(
            id=self.prefix('save_height_in'),
            placeholder=500,
            type='number',
            value=500,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.output_height_ig = self.make_input_group(
            [
                dbc.InputGroupText('Output height'),
                self.output_height_input,
                dbc.InputGroupText('px'),
            ]
        )

        self.output_width_input = dbc.Input(
            id=self.prefix('save_width_in'),
            placeholder=500,
            type='number',
            value=500,
            style={
                'textAlign': 'center',
                'verticalAlign': 'middle',
                'horizontalAlign': 'middle'
            }
        )

        self.output_width_ig = self.make_input_group(
            [
                dbc.InputGroupText('Output width'),
                self.output_width_input,
                dbc.InputGroupText('px'),
            ]
        )

        self.download_button = dbc.Button(
            'Download Data',
            id=self.prefix('download_data'),
            style={
                'boxShadow': 'none',
                'textalign': 'top'
            }
        )
        self.download_trigger = dcc.Download(
            id=self.prefix('download_data_trigger')
        )

        self.image_format_select = dbc.Select(
            id=self.prefix('save_format'),
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
            elements,
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
                            },
                        children='Orbital'
                    )
                ),
                dbc.Col(
                    html.H4(
                        style={
                            'textAlign': 'center',
                            },
                        children='Function'
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
                        self.func_select,
                        class_name='mb-3'
                    )
                ]
            ),
            html.H4(
                style={'textAlign': 'center'},
                children='Plot Options'
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        self.lower_x_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.upper_x_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        self.distance_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.colour_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row([
                dbc.Col([
                    html.H4(
                        style={
                            'textAlign': 'center',
                        },
                        children='Save Options',
                        id=self.prefix('save_options_header')
                    ),
                    dbc.Tooltip(
                        children='Use the camera button in the top right of \
                                the plot to save an image',
                        target=self.prefix('save_options_header'),
                        style={
                            'textAlign': 'center',
                        },
                    )
                ])
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        self.output_height_ig,
                        class_name='mb-3'
                    ),
                    dbc.Col(
                        self.output_width_ig,
                        class_name='mb-3'
                    )
                ]
            ),
            dbc.Row([
                dbc.Col(
                    [
                        self.download_button,
                        self.download_trigger
                    ],
                    class_name='mb-3'
                ),
                dbc.Col(
                    self.image_format_ig,
                    class_name='mb-3'
                )
            ])
        ]

        self.div.children = contents
        return


def assemble_2d_callbacks(plot_div: Orb2dPlot,
                          options_div: OptionsDiv) -> None:
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
        State(options_div.lower_x_input, 'value'),
        State(options_div.upper_x_input, 'value'),
        State(options_div.distance_select, 'value')
    ]
    callback(
        Output(options_div.download_trigger, 'data'),
        Input(options_div.download_button, 'n_clicks'),
        states
    )(download_data)


def download_data(_nc: int, func, orbs, low_x, up_x, unit):

    print(func, orbs, low_x, up_x, unit)

    return
