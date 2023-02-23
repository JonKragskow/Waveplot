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
import plotly.colors as pc
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, page_registry
import io
import base64
from dash.exceptions import PreventUpdate
import numpy as np
import sys
from ase import neighborlist, Atoms
from ase.geometry.analysis import Analysis
from dash.dependencies import Input, Output


# Dictionary of relative atomic masses
rams = {
    "H": 1.0076, "He": 4.0026, "Li": 6.941, "Be": 9.0122, "B": 10.811,
    "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984, "Ne": 20.1797,
    "Na": 22.9897, "Mg": 24.305, "Al": 26.9815, "Si": 28.0855, "P": 30.9738,
    "S": 32.065, "Cl": 35.453, "K": 39.0983, "Ar": 39.948, "Ca": 40.078,
    "Sc": 44.9559, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938,
    "Fe": 55.845, "Ni": 58.6934, "Co": 58.9332, "Cu": 63.546, "Zn": 65.39,
    "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96, "Br": 79.904,
    "Kr": 83.8, "Rb": 85.4678, "Sr": 87.62, "Y": 88.9059, "Zr": 91.224,
    "Nb": 92.9064, "Mo": 95.94, "Tc": 98, "Ru": 101.07, "Rh": 102.9055,
    "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818, "Sn": 118.71,
    "Sb": 121.76, "I": 126.9045, "Te": 127.6, "Xe": 131.293, "Cs": 132.9055,
    "Ba": 137.327, "La": 138.9055, "Ce": 140.116, "Pr": 140.9077, "Nd": 144.24,
    "Pm": 145, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.9253,
    "Dy": 162.5, "Ho": 164.9303, "Er": 167.259, "Tm": 168.9342, "Yb": 173.04,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.9479, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.078, "Au": 196.9665, "Hg": 200.59,
    "Tl": 204.3833, "Pb": 207.2, "Bi": 208.9804, "Po": 209, "At": 210,
    "Rn": 222, "Fr": 223, "Ra": 226, "Ac": 227, "Pa": 231.0359, "Th": 232.0381,
    "Np": 237, "U": 238.0289, "Am": 243, "Pu": 244, "Cm": 247, "Bk": 247,
    "Cf": 251, "Es": 252, "Fm": 257, "Md": 258, "No": 259, "Rf": 261,
    "Lr": 262, "Db": 262, "Bh": 264, "Sg": 266, "Mt": 268, "Rg": 272, "Hs": 277
}

lab_num = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
    "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
    "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
    "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
    "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97,
    "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103,
    "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108,
    "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113,
    "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

num_lab = dict(zip(lab_num.values(), lab_num.keys()))

# Atomic radii
# https://www.rsc.org/periodic-table/
atomic_radii = {
    "H": 1.1, "He": 1.4, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.7,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.1, "P": 1.8, "S": 1.8, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Sc": 2.15, "Ti": 2.11, "V": 2.07, "Cr": 2.06,
    "Mn": 2.05, "Fe": 2.04, "Co": 2.0, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.9, "Br": 1.85, "Kr": 2.02,
    "Rb": 3.03, "Sr": 2.49, "Y": 2.32, "Zr": 2.23, "Nb": 2.18, "Mo": 2.17,
    "Tc": 2.16, "Ru": 2.13, "Rh": 2.1, "Pd": 2.1, "Ag": 2.11, "Cd": 2.18,
    "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16,
    "Cs": 3.43, "Ba": 2.68, "La": 2.43, "Ce": 2.42, "Pr": 2.4, "Nd": 2.39,
    "Pm": 2.38, "Sm": 2.36, "Eu": 2.35, "Gd": 2.34, "Tb": 2.33, "Dy": 2.31,
    "Ho": 2.3, "Er": 2.29, "Tm": 2.27, "Yb": 2.26, "Lu": 2.24, "Hf": 2.23,
    "Ta": 2.22, "W": 2.18, "Re": 2.16, "Os": 2.16, "Ir": 2.13, "Pt": 2.13,
    "Au": 2.14, "Hg": 2.23, "Tl": 1.96, "Pb": 2.02, "Bi": 2.07, "Po": 1.97,
    "At": 2.02, "Rn": 2.2, "Fr": 3.48, "Ra": 2.83, "Ac": 2.47, "Th": 2.45,
    "Pa": 2.43, "U": 2.41, "Np": 2.39, "Pu": 2.43, "Am": 2.44, "Cm": 2.45,
    "Bk": 2.44, "Cf": 2.45, "Es": 2.45, "Fm": 2.45, "Md": 2.46, "No": 2.46,
    "Lr": 2.46
}
# Covalent radii
# https://www.rsc.org/periodic-table/
cov_radii = {
    "H": 0.32, "He": 0.37, "Li": 1.3, "Be": 0.99, "B": 0.84, "C": 0.75,
    "N": 0.71, "O": 0.64, "F": 0.6, "Ne": 0.62, "Na": 1.6, "Mg": 1.4,
    "Al": 1.24, "Si": 1.14, "P": 1.09, "S": 1.04, "Cl": 1.0, "Ar": 1.01,
    "K": 2.0, "Ca": 1.74, "Sc": 1.59, "Ti": 1.48, "V": 1.44, "Cr": 1.3,
    "Mn": 1.29, "Fe": 1.24, "Co": 1.18, "Ni": 1.17, "Cu": 1.22, "Zn": 1.2,
    "Ga": 1.23, "Ge": 1.2, "As": 1.2, "Se": 1.18, "Br": 1.17, "Kr": 1.16,
    "Rb": 2.15, "Sr": 1.9, "Y": 1.76, "Zr": 1.64, "Nb": 1.56, "Mo": 1.46,
    "Tc": 1.38, "Ru": 1.36, "Rh": 1.34, "Pd": 1.3, "Ag": 1.36, "Cd": 1.4,
    "In": 1.42, "Sn": 1.4, "Sb": 1.4, "Te": 1.37, "I": 1.36, "Xe": 1.36,
    "Cs": 2.38, "Ba": 2.06, "La": 1.94, "Ce": 1.84, "Pr": 1.9, "Nd": 1.88,
    "Pm": 1.86, "Sm": 1.85, "Eu": 1.83, "Gd": 1.82, "Tb": 1.81, "Dy": 1.8,
    "Ho": 1.79, "Er": 1.77, "Tm": 1.77, "Yb": 1.78, "Lu": 1.74, "Hf": 1.64,
    "Ta": 1.58, "W": 1.5, "Re": 1.41, "Os": 1.36, "Ir": 1.32, "Pt": 1.3,
    "Au": 1.3, "Hg": 1.32, "Tl": 1.44, "Pb": 1.45, "Bi": 1.5, "Po": 1.42,
    "At": 1.48, "Rn": 1.46, "Fr": 2.42, "Ra": 2.11, "Ac": 2.01, "Th": 1.9,
    "Pa": 1.84, "U": 1.83, "Np": 1.8, "Pu": 1.8, "Am": 1.73, "Cm": 1.68,
    "Bk": 1.68, "Cf": 1.68, "Es": 1.65, "Fm": 1.67, "Md": 1.73, "No": 1.76,
    "Lr": 1.61, "Rf": 1.57, "Db": 1.49, "Sg": 1.43, "Bh": 1.41, "Hs": 1.34,
    "Mt": 1.29, "Ds": 1.28, "Rg": 1.21, "Cn": 1.22, "Nh": 1.36, "Fl": 1.43,
    "Mc": 1.62, "Lv": 1.75, "Ts": 1.65, "Og": 1.57
}


atom_colours = {
    "H": "#999999",
    "Li": "#9932CC",
    "Na": "#9932CC",
    "K": "#9932CC",
    "Rb": "#9932CC",
    "Cs": "#9932CC",
    "Fr": "#9932CC",
    "Be": "#4E7566",
    "Mg": "#4E7566",
    "Ca": "#4E7566",
    "Sr": "#4E7566",
    "Ba": "#4E7566",
    "Ra": "#4E7566",
    "Sc": "#4E7566",
    "Y": "#4E7566",
    "La": "#4E7566",
    "Ac": "#4E7566",
    "Ti": "#301934",
    "V": "#301934",
    "Cr": "#301934",
    "Mn": "#301934",
    "Fe": "#301934",
    "Ni": "#301934",
    "Co": "#301934",
    "Cu": "#301934",
    "Zn": "#301934",
    "Zr": "#301943",
    "Nb": "#301943",
    "Mo": "#301943",
    "Tc": "#301943",
    "Ru": "#301943",
    "Rh": "#301943",
    "Pd": "#301943",
    "Ag": "#301943",
    "Cd": "#301943",
    "Hf": "#301934",
    "Ta": "#301934",
    "W": "#301934",
    "Re": "#301934",
    "Os": "#301934",
    "Ir": "#301934",
    "Pt": "#301934",
    "Au": "#301934",
    "Hg": "#301934",
    "B": "#FFFF00",
    "C": "#696969",
    "N": "#0000FF",
    "O": "#FF0000",
    "F": "#228B22",
    "Al": "#800080",
    "Si": "#FF7F50",
    "P": "#FF00FF",
    "S": "#FFFF00",
    "Cl": "#228B22",
    "As": "#F75394",
    "Br": "#4A2600",
    "other": "#3f3f3f"
}

tol_cols = [
    "rgb(0  , 0  , 0)",
    "rgb(230, 159, 0)",
    "rgb(86 , 180, 233)",
    "rgb(0  , 158, 115)",
    "rgb(240, 228, 66)",
    "rgb(0  , 114, 178)",
    "rgb(213, 94 , 0)",
    "rgb(204, 121, 167)"
]
# Bang wong list of colourblindness friendly colours
# https://www.nature.com/articles/nmeth.1618
wong_cols = [
    "rgb(51 , 34 , 136)",
    "rgb(17 , 119, 51)",
    "rgb(68 , 170, 153)",
    "rgb(136, 204, 238)",
    "rgb(221, 204, 119)",
    "rgb(204, 102, 119)",
    "rgb(170, 68 , 153)",
    "rgb(136, 34 , 85)"
]
# Default list of colours is plotly"s safe colourlist
def_cols = pc.qualitative.Safe


def dash_id(page: str):
    def func(_id: str):
        return f"{page}_{_id}"
    return func


def create_navbar(current_path):
    """
    Creates navbar element for current_page

    Parameters
    ----------
    current_page : str
        Name of webpage which navbar will appear on

    Returns
    -------
    dbc.NavbarSimple
        Navbar focussed on current page, with other pages included
    """

    paths = [
        page['path'] for page in page_registry.values()
    ]

    names = [
        page['name'] for page in page_registry.values()
    ]

    current_name = None
    for path, name in zip(paths, names):
        if current_path == path:
            current_name = name

    dropdown = [dbc.DropdownMenuItem("More pages", header=True)]

    for path, name in zip(paths, names):
        if path not in [current_path, '/']:
            dropdown.append(
                dbc.DropdownMenuItem(
                    name,
                    href=path
                )
            )

    # Icons for navbar
    # these are hyperlinked
    icons = dbc.Row(
        [
            dbc.Col(
                html.A(
                    html.I(
                        className="fab fa-github fa-2x",
                        style={
                            'color': 'white'
                        }
                    ),
                    href="https://github.com/jonkragskow/waveplot"
                )
            ),
            dbc.Col(
                html.A(
                    html.I(
                        className="fa-solid fa-question fa-2x",
                        style={
                            'color': 'white'
                        }
                    ),
                    href="/assets/waveplot_docs.pdf",
                )
            ),
            # dbc.Col(
            #     html.A(
            #         html.I(
            #             className="fa-solid fa-book fa-2x",
            #             style={
            #                 'color': 'white'
            #             }
            #         ),
            #         href="PAPER_URL"
            #     )
            # )
        ],
        style={
            'position': 'absolute',
            'right': '40px'
        },
        class_name='nav_buttons'
    )

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(current_name, href=current_path)),
            dbc.DropdownMenu(
                children=dropdown,
                nav=True,
                in_navbar=True,
                label="More",
            ),
            html.Div(icons)
        ],
        brand="Waveplot",
        brand_href="/",
        color="#307cff",
        dark=True,
        links_left=True,
    )
    return navbar


def footer(name="Jon Kragskow", site="https://www.kragskow.dev"):

    footer = html.Footer(
        className="footer_custom",
        children=[
            html.Div(
                [
                    dbc.Button(
                        name,
                        href=site,
                        style={
                            "color": "",
                            'backgroundColor': 'transparent',
                            'borderColor': 'transparent',
                            'boxShadow': 'none',
                            'textalign': 'top'
                        }
                    )
                ]
            )
        ],
        style={'whiteSpace': 'pre-wrap'}
    )

    return footer


def load_xyz(f_name: str):
    """
    Load labels and coordinates from a .xyz file

    Parameters
    ----------
    f_name : str
        File name
    atomic_numbers : bool, default False
        If True, reads xyz file with atomic numbers and converts to labels
    add_indices : bool, default False
        If True, add indices to atomic labels
        (replacing those which may exist already)
    capitalise : bool, default True
        If True, capitalise atomic labels

    Returns
    -------
    list
        atomic labels
    np.ndarray
        (n_atoms,3) array containing xyz coordinates of each atom
    """

    _labels = np.loadtxt(f_name, skiprows=2, usecols=0, dtype=str, ndmin=1)
    _labels = _labels.tolist()
    _labels = [lab.capitalize() for lab in _labels]

    _coords = np.loadtxt(f_name, skiprows=2, usecols=(1, 2, 3), ndmin=2)

    return _labels, _coords


def parse_xyz_file(xyz_file):
    """
    Read in xyz file from either local server or from file provided as encoded
    string

    Parameters
    ----------
    xyz_file : str
        String containing file name or binary encoded file contents

    Returns
    -------
    list
        Atomic labels
    np.ndarray
        Atomic coordinates as (n_atoms,3) array

    Raises
    ------
    PreventUpdate
        if xyz_file is empty
    """

    # Local default just load
    if ".xyz" in xyz_file:
        labels, coords = load_xyz(xyz_file)
    # Read binary encoded string containing file, decode and then read as
    # stringio object
    else:
        if len(xyz_file.split(",")) == 1:
            raise PreventUpdate
        _, content_string = xyz_file.split(",")
        decoded = base64.b64decode(content_string).decode('utf8')
        xyz_file_str = io.StringIO(decoded)
        labels = np.loadtxt(xyz_file_str, skiprows=2, usecols=(0), dtype=str)
        xyz_file_str.seek(0)
        coords = np.loadtxt(xyz_file_str, skiprows=2, usecols=(1, 2, 3))

    return labels, coords


def make_js_molecule(coords, labels_nn, adjacency):
    """
    Creates dictionaries used to add molecules to 3Dmol.js viewer

    Resulting list of dictionaries is used as sole argument of js call to
    m.addAtoms();
    where m is defined as output of
    viewer.addModel();

    Parameters
    ----------
    coords : np.ndarray
        [n_atoms,3] array containing x, y, z coordinates of each atom
    labels : list
        Atom labels without indexing
    adjacency : np.ndarray
        [n_atoms, n_atoms] Adjacency matrix used to define bonds between atoms

    Returns
    -------
    list:
        dictionaries, one per atom, describing atom label, position, and
        indices of other atoms bonded to that atom

    """

    n_bonds = np.sum(adjacency, axis=1)

    molecule = []
    for it, (xyz, label) in enumerate(zip(coords, labels_nn)):
        atdict = {}
        atdict["elem"] = label
        atdict["x"] = xyz[0]
        atdict["y"] = xyz[1]
        atdict["z"] = xyz[2]
        atdict["bonds"] = [
            jt for (jt, oz) in enumerate(adjacency[it]) if oz
        ]
        atdict["bondOrder"] = [1]*n_bonds[it]
        molecule.append(atdict)

    return molecule


def make_js_label(coords, labels, viewer_var):
    """
    Creates javascript code to display atom labels

    Parameters
    ----------
    coords : np.ndarray
        [n_atoms,3] array containing x, y, z coordinates of each atom
    labels : list
        Atom labels without indexing
    viewer_var : str
        Name of 3Dmol.js viewer variable defined as output of
        $3Dmol.createViewer();

    Returns
    -------
    str:
        String containing one viewer_var.addLabel() call for each atom

    """

    atom_labels = ''

    for coord, label in zip(coords, labels):
        atom_labels += '\n{}.addLabel(\n'.format(viewer_var)
        atom_labels += '    "{}",\n'.format(label)
        atom_labels += '    {position:'
        atom_labels += ' {x:'
        atom_labels += '{:f}'.format(coord[0])
        atom_labels += ', y:'
        atom_labels += '{:f}'.format(coord[1])
        atom_labels += ', z:'
        atom_labels += '{:f}'.format(coord[2])
        atom_labels += '},\n'
        atom_labels += '  inFront: true}'
        atom_labels += ');\n'

    return atom_labels


def make_js_molstyle(style, labels_nn, radius_type, model_var):
    """
    Creates javascript code (as string) to set atom display style.

    Parameters
    ----------
    style : str {"stick", "sphere"}
        Set radius type used to draw atoms
    labels_nn : list
        Atomic labels without indexing
    radius_type : str {"covalent", "atomic"}
        Set size of "sphere" used in "sphere" representation of atoms
    model_var : str
        Name of 3Dmol.js model variable in js codedefined as output of
        viewer.addModel()

    Returns
    -------
    str:
        String containing m.setStyle() calls, one for each element type

    """

    unique_labs = np.unique(labels_nn)

    # Set radius of each atom
    if radius_type == "covalent":
        radii = [cov_radii[lab] for lab in unique_labs]
    elif radius_type == "atomic":
        radii = [atomic_radii[lab] for lab in unique_labs]

    # Generate 3Dmol.js setStyle(sel, style) code as string
    if style == 'stick':
        molstyle_str = "{}.setStyle(".format(model_var)
        molstyle_str += "{},{stick:{}});"
    elif style == 'sphere':
        molstyle_str = ''
        for lab, rad in zip(unique_labs, radii):
            molstyle_str += '{}'.format(model_var)
            molstyle_str += '.setStyle({'
            molstyle_str += "elem:'{}'".format(lab)
            molstyle_str += '},{sphere:{'
            molstyle_str += 'radius:{:f}'.format(rad) + '}'
            molstyle_str += '});\n'

    return molstyle_str


def make_js_cylinder(end_coords, color, viewer_var, width=1., scale=1.,
                     start_coords=[0., 0., 0.]):
    """
    Creates javascript code to display a single cylinder

    Parameters
    ----------
    end_coords : np.ndarray
        [3] array containing x, y, z coordinates of cylinder end point
    color : str
        Colour of cylinder as hex
    viewer_var : str
        Name of 3Dmol.js viewer variable defined as output of
        $3Dmol.createViewer();
    width : float, defauult 1.
        Radius of cylinder
    scale : float, default 1.
        scaling factor to apply to cylinder length
    start_coords : np.ndarray, default [0.,0.,0.] = origin
        [3] array containing x, y, z coordinates of cylinder start point

    Returns
    -------
    str:
        String containing one viewer.addCylinder call

    """

    cylinder = '{}'.format(viewer_var)
    cylinder += '.addCylinder({\n'
    cylinder += '    start: {x:'
    cylinder += '{:f}'.format(start_coords[0]*scale)
    cylinder += ', y:'
    cylinder += '{:f}'.format(start_coords[1]*scale)
    cylinder += ', z:'
    cylinder += '{:f}'.format(start_coords[2]*scale)
    cylinder += '},\n'
    cylinder += '    end: {x:'
    cylinder += '{:f}'.format(end_coords[0]*scale)
    cylinder += ', y:'
    cylinder += '{:f}'.format(end_coords[1]*scale)
    cylinder += ', z:'
    cylinder += '{:f}'.format(end_coords[2]*scale)
    cylinder += '},\n'
    cylinder += '    radius: {:f},\n'.format(width)
    cylinder += '    toCap: 1,\n'
    cylinder += '    fromCap: 2,\n'
    cylinder += "    color: '{}',\n".format(color)
    cylinder += '});\n'

    return cylinder


def make_js_sphere(coords, color, viewer_var, scale=1.):
    """
    Creates javascript code to display a single sphere

    Parameters
    ----------
    coords : np.ndarray
        [3] array containing x, y, z coordinates of sphere centre
    color : str
        Colour of cylinder as hex
    scale : float, default 1.
        scaling factor to apply to sphere centre
    viewer_var : str
        Name of 3Dmol.js viewer variable defined as output of
        $3Dmol.createViewer();

    Returns
    -------
    str:
        String containing one viewer.addSphere call

    """

    sphere = '{}'.format(viewer_var)
    sphere += '.addSphere({\n'
    sphere += '    center: {x:'
    sphere += '{:f}'.format(coords[0]*scale)
    sphere += ', y:'
    sphere += '{:f}'.format(coords[1]*scale)
    sphere += ', z:'
    sphere += '{:f}'.format(coords[2]*scale)
    sphere += '},\n'
    sphere += '    radius: 0.1,\n'
    sphere += "    color: '{}',\n".format(color)
    sphere += '});\n'

    return sphere


def remove_label_indices(labels):
    """
    Remove label indexing from atomic symbols
    indexing is either numbers or numbers followed by letters:
    e.g. H1, H2, H3
    or H1a, H2a, H3a

    Parameters
    ----------
    labels : list
        atomic labels

    Returns
    -------
    list
        atomic labels without indexing
    """

    labels_nn = []
    for label in labels:
        no_digits = []
        for i in label:
            if not i.isdigit():
                no_digits.append(i)
            elif i.isdigit():
                break
        result = ''.join(no_digits)
        labels_nn.append(result)

    return labels_nn


def add_label_indices(labels, style='per_element'):
    """
    Add label indexing to atomic symbols - either element or per atom.

    Parameters
    ----------
    labels : list
        atomic labels
    style : str, optional
        {'per_element', 'sequential'}
            'per_element' : Index by element e.g. Dy1, Dy2, N1, N2, etc.
            'sequential' : Index the atoms 1->N regardless of element

    Returns
    -------
    list
        atomic labels with indexing
    """

    # remove numbers just in case
    labels_nn = remove_label_indices(labels)

    # Just number the atoms 1->N regardless of element
    if style == 'sequential':
        labels_wn = ['{}{:d}'.format(lab, it+1)
                     for (it, lab) in enumerate(labels)]

    # Index by element Dy1, Dy2, N1, N2, etc.
    if style == 'per_element':
        # Get list of unique elements
        atoms = set(labels_nn)
        # Create dict to keep track of index of current atom of each element
        atom_count = {atom: 1 for atom in atoms}
        # Create labelled list of elements
        labels_wn = []
        for lab in labels_nn:
            # Index according to dictionary
            labels_wn.append("{}{:d}".format(lab, atom_count[lab]))
            # Then add one to dictionary
            atom_count[lab] += 1

    return labels_wn


def process_labels_coords(labels, coords):
    """
    Sorts labels and coordinates by mass (heaviest first), and reindexes labels
    accordingly
    """

    labels_nn = remove_label_indices(labels)

    # Sort by mass (heaviest first)
    masses = [rams[lab] for lab in labels_nn]
    order = np.argsort(masses)[::-1]

    labels_nn = [labels_nn[orde] for orde in order]
    labels = add_label_indices(labels_nn)
    coords = coords[order]

    return labels, labels_nn, coords


def get_neighborlist(labels, coords, adjust_cutoff={}):
    """
    Calculate ASE neighbourlist based on covalent radii

    Parameters
    ----------
    labels : list
        Atomic labels
    coords : np.ndarray
        xyz coordinates as (n_atoms, 3) array
    adjust_cutoff : dict, optional
        dictionary of atoms (keys) and new cutoffs (values)

    Returns
    -------
    ASE neighbourlist object
        Neighbourlist for system
    """

    # Remove labels if present
    labels_nn = remove_label_indices(labels)

    # Load molecule
    mol = Atoms("".join(labels_nn), positions=coords)

    # Define cutoffs for each atom using atomic radii
    cutoffs = neighborlist.natural_cutoffs(mol)

    # Modify cutoff if requested
    if adjust_cutoff:
        for it, label in enumerate(labels_nn):
            if label in adjust_cutoff.keys():
                cutoffs[it] = adjust_cutoff[label]

    # Create neighbourlist using cutoffs
    neigh_list = neighborlist.NeighborList(
        cutoffs=cutoffs,
        self_interaction=False,
        bothways=True
    )

    # Update this list by specifying the atomic positions
    neigh_list.update(mol)

    return neigh_list


def get_adjacency(labels, coords, adjust_cutoff={}):
    """
    Calculate adjacency matrix using ASE based on covalent radii.

    Parameters
    ----------
    labels : list
        Atomic labels
    coords : np.ndarray
        xyz coordinates as (n_atoms, 3) array
    adjust_cutoff : dict, optional
        dictionary of atoms (keys) and new cutoffs (values)
    save : bool, default False
        If true save to file given by `f_name`
    f_name : str, default 'adjacency.dat'
        If save true, this name is used for the file containing the adjacency
        matrix

    Returns
    -------
    np.array
        Adjacency matrix with same order as labels/coords
    """

    # Remove labels if present
    labels_nn = remove_label_indices(labels)

    # Get ASE neighbourlist object
    neigh_list = get_neighborlist(labels_nn, coords,
                                  adjust_cutoff=adjust_cutoff)

    # Create adjacency matrix
    adjacency = neigh_list.get_connectivity_matrix(sparse=False)

    return adjacency


def get_bonds(labels, coords, neigh_list=None, verbose=True, style='indices'):
    """
    Calculate list of atoms between which there is a bond.
    Using ASE. Only unique bonds are retained.
    e.g. 0-1 and not 1-0

    Parameters
    ----------
    labels : list
        Atomic labels
    coords : np.ndarray
        xyz coordinates as (n_atoms, 3) array
    neigh_list : ASE neighbourlist object, optional
        neighbourlist of system
    f_name : str, 'bonds.dat'
        filename to save bond list to
    save : bool, default False
        Save bond list to file
    verbose : bool, default True
        Print number of bonds to screen
    style : str, {'indices','labels'}
            indices : Bond list contains atom number
            labels  : Bond list contains atom label

    Returns
    -------
    list
        list of lists of unique bonds (atom pairs)
    """

    # Remove labels if present
    labels_nn = remove_label_indices(labels)

    # Create molecule object
    mol = Atoms("".join(labels_nn), positions=coords)

    # Get neighbourlist if not provided to function
    if not neigh_list:
        neigh_list = get_neighborlist(labels, coords)

    # Get object containing analysis of molecular structure
    ana = Analysis(mol, nl=neigh_list)

    # Get bonds from ASE
    # Returns: list of lists of lists containing UNIQUE bonds
    # Defined as
    # Atom 1 : [bonded atom, bonded atom], ...
    # Atom 2 : [bonded atom, bonded atom], ...
    # Atom n : [bonded atom, bonded atom], ...
    # Where only the right hand side is in the list
    is_bonded_to = ana.unique_bonds

    # Remove weird outer list wrapping the entire thing twice...
    is_bonded_to = is_bonded_to[0]
    # Create list of bonds (atom pairs) by appending lhs of above
    # definition to each element of the rhs
    bonds = []
    for it, ibt in enumerate(is_bonded_to):
        for atom in ibt:
            bonds.append([it, atom])

    # Count bonds
    n_bonds = len(bonds)

    # Set format and convert to atomic labels if requested
    if style == "labels":
        bonds = [
            [labels[atom1], labels[atom2]]
            for atom1, atom2 in bonds
        ]
    elif style == "indices":
        pass
    else:
        sys.exit("Unknown style specified")

    # Print number of bonds to screen
    if verbose:
        print('{:d}'.format(n_bonds)+' bonds')

    return bonds


def create_bond_choices(labels, coords):

    bond_labels = get_bonds(
        labels, coords, style="labels", verbose=False
    )
    bond_inds = get_bonds(labels, coords, style="indices", verbose=False)

    bond_choices = [
        {
            "label": "{}-{}".format(*bl),
            "value": [int(bi[0]), int(bi[1])]
        } for bl, bi in zip(bond_labels, bond_inds)
    ]

    return bond_choices


def define_centres(labels, coords, centre_label):
    """
    Creates list of potential centre positions for ray tracing
    using atomic labels and positions, and returns chosen centre based
    on specified (indexed) atomic label

    Parameters
    ----------
    labels : list
        Atomic labels
    coords : np.ndarray
        Atomic coordinates as (n_atoms,3) array
    centre_label : str
        Indexed atomic label used to identify centre e.g. Dy1

    Returns
    -------
    dict
        keys =  "label", "value"
        values =  atomic labels as strings
        Specifies potential choices for centre
    int
        Index of centre in coords array
    np.ndarray
        Position vector corresponding to centre
    """

    # List of possible centres
    centre_choices = [
        {"label": lab, "value": lab} for lab in labels
    ]

    centre_index = labels.index(centre_label)
    centre_coords = coords[centre_index, :]

    return centre_choices, centre_index, centre_coords
