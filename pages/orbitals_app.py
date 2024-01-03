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

from .core import orbitals as oc
from .core import common as com

ID_PREFIX = 'orb'
dash_id = com.dash_id(ID_PREFIX)

PAGE_NAME = 'Atomic Orbitals'
PAGE_PATH = '/orbitals'
PAGE_IMAGE = 'assets/orbital.png'
PAGE_DESCRIPTION = 'Interactive atomic orbitals'

register_page(
    __name__,
    order=4,
    path=PAGE_PATH,
    name=PAGE_NAME,
    title=PAGE_NAME,
    image=PAGE_IMAGE,
    description=PAGE_DESCRIPTION
)

default_orb = '3d+0'

plot_div = com.PlotDiv(
    ID_PREFIX,
    layout=oc.ORB_LAYOUT,
    config=oc.ORB_CONFIG,
    plot_loading=True,
    store_data=default_orb
)

# Make AC options tab and all callbacks
options = oc.OptionsDiv(ID_PREFIX, default_orb=default_orb)
# Connect callbacks for plots and options
oc.assemble_callbacks(plot_div, options)

# Layout of webpage
layout = com.make_layout(plot_div.div, options.div)
