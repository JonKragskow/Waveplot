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

from dash import register_page
from .core import utils
from .core import vibrations as vib
from .core import common as com

ID_PREFIX = 'vib'
dash_id = com.dash_id(ID_PREFIX)

PAGE_NAME = 'Vibrations'
PAGE_PATH = '/vibrations'
PAGE_IMAGE = 'assets/Vibrations.png'
PAGE_DESCRIPTION = 'Interactive harmonic oscillator wavefunctions and energy levels' # noqa

register_page(
    __name__,
    order=2,
    path=PAGE_PATH,
    name=PAGE_NAME,
    title=PAGE_NAME,
    image=PAGE_IMAGE,
    description=PAGE_DESCRIPTION
)

plot_div = com.PlotDiv(ID_PREFIX, vib.VIB_LAYOUT, com.BASIC_CONFIG)

# Make AC options tab and all callbacks
options = vib.OptionsDiv(ID_PREFIX)
# Connect callbacks for plots and options
vib.assemble_callbacks(plot_div, options)

# Layout of webpage
layout = com.make_layout(plot_div.div, options.div)
