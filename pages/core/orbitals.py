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
    '1s+0': 0.1,
    '2s+0': 0.01,
    '3s+0': 0.001,
    '4s+0': 0.001,
    '5s+0': 0.001,
    '6s+0': 0.0005,
    '2p+0': 0.01,
    '3p+0': 0.001,
    '4p+0': 0.001,
    '5p+0': 0.001,
    '6p+0': 0.0006,
    '2p-1': 0.01,
    '3p-1': 0.001,
    '4p-1': 0.001,
    '5p-1': 0.001,
    '6p-1': 0.0006,
    '2p+1': 0.01,
    '3p+1': 0.001,
    '4p+1': 0.001,
    '5p+1': 0.001,
    '6p+1': 0.0006,
    '3d+0': 0.09,
    '4d+0': 0.09,
    '5d+0': 0.1,
    '6d+0': 0.1,
    '3d+2': 0.09,
    '4d+2': 0.09,
    '5d+2': 0.01,
    '6d+2': 0.01,
    '3d-2': 0.09,
    '4d-2': 0.09,
    '5d-2': 0.01,
    '6d-2': 0.01,
    '3d-1': 0.09,
    '4d-1': 0.09,
    '5d-1': 0.01,
    '6d-1': 0.01,
    '3d+1': 0.09,
    '4d+1': 0.09,
    '5d+1': 0.01,
    '6d+1': 0.01,
    '4f+0': 0.0005,
    '4f+2c': 0.0005,
    '4f-2c': 0.0005,
    '5f+0': 0.0005,
    '5f+2c': 0.0005,
    '5f-2c': 0.0005,
    '6f+0': 0.0004,
    '6f+2c': 0.0004,
    '6f-2c': 0.0004,
    '4f-2': 0.0006,
    '5f-2': 0.0006,
    '6f-2': 0.0004,
    '4f+2': 0.0006,
    '5f+2': 0.0006,
    '6f+2': 0.0004,
    '4f+3c': 0.0006,
    '5f+3c': 0.0006,
    '6f+3c': 0.0004,
    '4f-3c': 0.0006,
    '5f-3c': 0.0006,
    '6f-3c': 0.0004,
    '4f-3': 0.0006,
    '5f-3': 0.0006,
    '6f-3': 0.0004,
    '4f+3': 0.0006,
    '5f+3': 0.0006,
    '6f+3': 0.0004,
    '4f-1': 0.0006,
    '4f+1': 0.0006,
    '5f-1': 0.0006,
    '5f+1': 0.0006,
    '6f-1': 0.0006,
    '6f+1': 0.0006,
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
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
    '''

    if n == 1:
        bound = 5.
        step = 0.1
    elif n == 2:
        bound = 30.
        step = 0.5
    elif n == 3:
        bound = 40.
        step = 0.5
    elif n == 4:
        bound = 40.
        step = 0.5
    elif n == 5:
        bound = 60.
        step = 1.
    elif n == 6:
        bound = 80.
        step = 1.

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


def p_3d(n: int, ml: int, half: str = ''):
    '''
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    ml: int
        magnetic quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
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
    rad = rc.radial_p(n, 2 * r / n)

    # angular wavefunction
    if ml == 0:
        ang = np.sqrt(3. / (4. * np.pi)) * z / r
    elif ml == 1:
        ang = np.sqrt(3. / (4. * np.pi)) * y / r
    elif ml == -1:
        ang = np.sqrt(3. / (4. * np.pi)) * x / r
    else:
        raise ValueError('Incorrect ml value in p_3d')
    wav = ang * rad

    wav = np.nan_to_num(wav, 0, posinf=0., neginf=0.)

    return wav

    # if n == 3:
    #     bound = 70.
    #     step = 1.
    # elif n == 4:
    #     bound = 90.
    #     step = 2.
    # elif n == 5:
    #     bound = 110.
    #     step = 2.
    # elif n == 6:
    #     bound = 140.
    #     step = 2.


def d_3d(n: int, ml: int, half: str = ''):
    '''
    Calculates d orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    ml: int
        magnetic quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
    '''

    if n == 3:
        bound = 60.
        step = 1.
    elif n == 4:
        bound = 80.
        step = 1.
    elif n == 5:
        bound = 110.
        step = 2.
    elif n == 6:
        bound = 150.
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

    if ml == 0:
        ang = 3 * z**2 - r**2
    elif ml == -1:
        ang = x * z
    elif ml == 1:
        ang = y * z
    elif ml == -2:
        ang = x * y
    elif ml == 2:
        ang = x ** 2 - y ** 2
    else:
        raise ValueError('Incorrect ml value in d_3d')

    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad * ang

    return wav


def f_3d(n: int, ml: int, half: str = '', cubic: bool = False):
    '''
    Calculates f orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    ml: int
        magnetic quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.
    cubic: bool, default False
        If True, then cubic orbitals are calculated

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    if ml == 0:
        ang = 0.25 * np.sqrt(7 / np.pi) * (5 * z**3 - 3 * z * r**2) / r ** 3
    elif ml == -1:
        ang = 0.25 * np.sqrt(21 / (2 * np.pi)) * x * (5 * z**2 - r**2) / r**3
    elif ml == 1:
        ang = 0.25 * np.sqrt(21 / (2 * np.pi)) * y * (5 * z**2 - r**2) / r**3
    elif ml == -2:
        if cubic:
            ang = 0.25 * np.sqrt(105 / np.pi) * (z**2 - y**2) * x / (r**3)
        else:
            ang = 0.25 * np.sqrt(105 / np.pi) * x * y * z / (r**3)
    elif ml == 2:
        if cubic:
            ang = 0.25 * np.sqrt(105 / np.pi) * (z**2 - x**2) * y / (r**3)
        else:
            ang = 0.25 * np.sqrt(105 / np.pi) * (x**2 - y**2) * z / (r**3)
    elif ml == -3:
        if cubic:
            ang = 0.25 * np.sqrt(7 / np.pi) * (5 * x**3 - 3 * x * r**2) / r ** 3 # noqa
        else:
            ang = 0.25 * np.sqrt(35 / (2 * np.pi)) * (x**2 - 3 * y**2) * x / (r**3) # noqa
    elif ml == 3:
        if cubic:
            ang = 0.25 * np.sqrt(7 / np.pi) * (5 * y**3 - 3 * y * r**2) / r ** 3 # noqa
        else:
            ang = 0.25 * np.sqrt(35 / (2 * np.pi)) * (3 * x**2 - y**2) * y / (r**3) # noqa
    else:
        raise ValueError('Incorrect ml value in f_3d')

    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)
    wav = rad * ang

    return wav


def sp_3d(half: str):
    '''
    Calculates three-dimensional sp orbital on a grid of x, y, and z points\n

    Parameters
    ----------
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
    '''
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
    '''
    Calculates three-dimensional sp3 orbital on a grid of x, y, and z points\n

    Parameters
    ----------
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
    '''

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
    '''
    Calculates three-dimensional sp2 orbital on a grid of x, y, and z points\n

    Parameters
    ----------
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
    '''
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
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(105 / np.pi) * x * y * z / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fy3x2my2_3d(n: int, half: str = ''):
    '''
    Calculates fy(3x2-y2) orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(35 / (2 * np.pi)) * (3 * x**2 - y**2) * y / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fxx2m3y2_3d(n: int, half: str = ''):
    '''
    Calculates fx(x2-3y2) orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(35 / (2 * np.pi)) * (x**2 - 3 * y**2) * x / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fyz2mx2_3d(n: int, half: str = ''):
    '''
    Calculates fy(z2-x2) orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(105 / np.pi) * (z**2 - x**2) * y / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fxz2my2_3d(n: int, half: str = ''):
    '''
    Calculates fx(z2-y2) orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(105 / np.pi) * (z**2 - y**2) * x / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


def fzx2my2_3d(n: int, half: str = ''):
    '''
    Calculates fz(x2-y2) orbital wavefunction on a grid

    Parameters
    ----------
    n: int
        principal quantum number of orbital
    half: str {'', 'x', 'y', 'z'}
        Truncates x, y, or z at zero to create cross section of orbital data\n
        If empty then no truncation is performed.

    Returns
    -------
    ndarray of floats
        Meshgrid containing wavefunction
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

    ang = 0.25 * np.sqrt(105 / np.pi) * (x**2 - y**2) * z / (r**3)
    ang = np.nan_to_num(ang, 0, posinf=0., neginf=0.)

    wav = rad * ang

    return wav


class OptionsDiv(com.Div):
    def __init__(self, prefix, default_orb='3d+0', **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.orb_select = dbc.Select(
            id=self.prefix('orbital_select'),
            options=[
                {'label': '1s', 'value': '1s+0'},
                {'label': '2s', 'value': '2s+0'},
                {'label': '3s', 'value': '3s+0'},
                {'label': '4s', 'value': '4s+0'},
                {'label': '5s', 'value': '5s+0'},
                {'label': '6s', 'value': '6s+0'},
                {'label': '2px', 'value': '2p+1'},
                {'label': '2py', 'value': '2p-1'},
                {'label': '2pz', 'value': '2p+0'},
                {'label': '3px', 'value': '3p+1'},
                {'label': '3py', 'value': '3p-1'},
                {'label': '3pz', 'value': '3p+0'},
                {'label': '4px', 'value': '4p+1'},
                {'label': '4py', 'value': '4p-1'},
                {'label': '4pz', 'value': '4p+0'},
                {'label': '5px', 'value': '5p+1'},
                {'label': '5py', 'value': '5p-1'},
                {'label': '5pz', 'value': '5p+0'},
                {'label': '6px', 'value': '6p+1'},
                {'label': '6py', 'value': '6p-1'},
                {'label': '6pz', 'value': '6p+0'},
                {'label': '3dz²', 'value': '3d+0'},
                {'label': '3dxz', 'value': '3d+1'},
                {'label': '3dyz', 'value': '3d-1'},
                {'label': '3dx²-y²', 'value': '3d+2'},
                {'label': '3dxy', 'value': '3d-2'},
                {'label': '4dz²', 'value': '4d+0'},
                {'label': '4dxz', 'value': '4d+1'},
                {'label': '4dyz', 'value': '4d-1'},
                {'label': '4dx²-y²', 'value': '4d+2'},
                {'label': '4dxy', 'value': '4d-2'},
                {'label': '5dz²', 'value': '5d+0'},
                {'label': '5dxz', 'value': '5d+1'},
                {'label': '5dyz', 'value': '5d-1'},
                {'label': '5dx²-y²', 'value': '5d+2'},
                {'label': '5dxy', 'value': '5d-2'},
                {'label': '5dz²', 'value': '5d+0'},
                {'label': '6dz²', 'value': '6d+0'},
                {'label': '6dxz', 'value': '6d+1'},
                {'label': '6dyz', 'value': '6d-1'},
                {'label': '6dx²-y²', 'value': '6d+2'},
                {'label': '6dxy', 'value': '6d-2'},
                {'label': '4fz³', 'value': '4f+0'},
                {'label': '4fyz²', 'value': '4f-1'},
                {'label': '4fxz²', 'value': '4f+1'},
                {'label': '4fxyz', 'value': '4f-2'},
                {'label': '4fz(x²-y²)', 'value': '4f+2'},
                {'label': '4fy(3x²-y²)', 'value': '4f-3'},
                {'label': '4fx(x²-3y²)', 'value': '4f+3'},
                {'label': '4fx³ (cubic)', 'value': '4f+3c'},
                {'label': '4fy³ (cubic)', 'value': '4f-3c'},
                {'label': '4fy(z²-x²) (cubic)', 'value': '4f-2c'},
                {'label': '4fx(z²-y²) (cubic)', 'value': '4f+2c'},
                {'label': '5fz³', 'value': '5f+0'},
                {'label': '5fyz²', 'value': '5f-1'},
                {'label': '5fxz²', 'value': '5f+1'},
                {'label': '5fxyz', 'value': '5f-2'},
                {'label': '5fz(x²-y²)', 'value': '5f+2'},
                {'label': '5fy(3x²-y²)', 'value': '5f-3'},
                {'label': '5fx(x²-3y²)', 'value': '5f+3'},
                {'label': '5fx³ (cubic)', 'value': '5f+3c'},
                {'label': '5fy³ (cubic)', 'value': '5f-3c'},
                {'label': '5fy(z²-x²) (cubic)', 'value': '5f-2c'},
                {'label': '5fx(z²-y²) (cubic)', 'value': '5f+2c'},
                {'label': '6fz³', 'value': '6f+0'},
                {'label': '6fyz²', 'value': '6f-1'},
                {'label': '6fxz²', 'value': '6f+1'},
                {'label': '6fxyz', 'value': '6f-2'},
                {'label': '6fz(x²-y²)', 'value': '6f+2'},
                {'label': '6fy(3x²-y²)', 'value': '6f-3'},
                {'label': '6fx(x²-3y²)', 'value': '6f+3'},
                {'label': '6fx³ (cubic)', 'value': '6f+3c'},
                {'label': '6fy³ (cubic)', 'value': '6f-3c'},
                {'label': '6fy(z²-x²) (cubic)', 'value': '6f-2c'},
                {'label': '6fx(z²-y²) (cubic)', 'value': '6f+2c'},
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
            value=DEFAULT_ISO[default_orb],
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
            value='#8fbbd9',
            style={
                'height': '40px'
            }
        )

        self.colour_input_b = dbc.Input(
            id=str(uuid.uuid1()),
            type='color',
            value='#eb9393',
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
                        className='mb-3 text-center mwmob',
                        sm=12
                    )
                ],
                justify='center'
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        self.colours_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        self.axes_ig,
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
                        self.half_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    ),
                    dbc.Col(
                        self.isoval_ig,
                        class_name='mb-3 text-center mwmob',
                        sm=12,
                        md=6
                    )
                ],
                justify='center'
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
        'd': d_3d,
        'f': f_3d,
        'sp': sp_3d,
        'sp2': sp2_3d,
        'sp3': sp3_3d
    }

    special = ['sp', 'sp2', 'sp3']

    if orbital_name not in special:

        n = int(orbital_name[0])
        name = orbital_name[1]
        ml = int(orbital_name[2:4])

        # Cubic f
        if len(orbital_name) == 5:
            wav = orb_func_dict[name](
                n, ml, half, True
            )
        # s orbitals (ml always zero)
        elif name == 's':
            wav = orb_func_dict[name](
                n, half
            )
        # everything else
        else:
            wav = orb_func_dict[name](
                n, ml, half
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
    if 's' in orb_name and 'p' not in orb_name:
        rounds = 0
    else:
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
        if orb_name == '1s+0':
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
    if orb_name != '1s+0':
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
