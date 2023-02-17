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

from dash import html, dcc, callback_context, no_update, register_page,\
    callback, clientside_callback

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import numpy.linalg as la

from .core import sievers
from .core import utils

ID_PREFIX = "aniso"

id = utils.dash_id(ID_PREFIX)

PAGE_NAME = '4f Densities'
PAGE_PATH = '/f-densities'
PAGE_IMAGE = 'assets/4f_densities.png'
PAGE_DESCRIPTION = "Interactive lanthanide free-ion densities"

register_page(
    __name__,
    order=3,
    path=PAGE_PATH,
    name=PAGE_NAME,
    title=PAGE_NAME,
    image=PAGE_IMAGE,
    description=PAGE_DESCRIPTION
)

# Load default xyz file and get bonds for z axis
default_xyz = "assets/nature.xyz"

labels, coords = utils.load_xyz(default_xyz)
labels, labels_nn, coords = utils.process_labels_coords(labels, coords)
default_bonds = utils.create_bond_choices(labels, coords)

"""
Webpage layout
"""
surface_options = [
    html.Div(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H4(
                        style={"textAlign": "center"},
                        children="Parameters",
                        className="mb-3"
                    )
                )
            ),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "# f Electrons"
                            ),
                            dbc.Input(
                                id=id('n_value'),
                                placeholder="9",
                                type="number",
                                value=9,
                                style={
                                    "textAlign": "center"
                                },
                                min=1,
                                max=13
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "J"
                            ),
                            dbc.Input(
                                id=id('J_value'),
                                placeholder="7.5",
                                type="number",
                                value=7.5,
                                style={
                                    "textAlign": "center"
                                },
                                min=0.5
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "mJ"
                            ),
                            dbc.Input(
                                id=id('mJ_value'),
                                placeholder="7.5",
                                type="number",
                                value=7.5,
                                style={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ])
            ]),
            dbc.Row(
                id=id("L_S_wrapper"),
                children=[
                    dbc.Col([
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    "L"
                                ),
                                dbc.Input(
                                    id=id('L_value'),
                                    placeholder="5",
                                    type="number",
                                    value=5,
                                    style={
                                        "textAlign": "center"
                                    }
                                )
                            ],
                            class_name="mb-3",
                        ),
                    ]),
                    dbc.Col([
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    "S"
                                ),
                                dbc.Input(
                                    id=id('S_value'),
                                    placeholder="2.5",
                                    type="number",
                                    value=2.5,
                                    style={
                                        "textAlign": "center"
                                    },
                                    min=0.5
                                )
                            ],
                            class_name="mb-3",
                        ),
                    ]),
                ],
                style={"display": "none"}
            )
        ]
    )
]

spheroid_options = [
    html.Div(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H5(
                        style={"textAlign": "center"},
                        children="Surface",
                        className="mb-3"
                    )
                )
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Surface"
                                    ),
                                    dbc.InputGroupText(
                                        dbc.Checkbox(
                                            value=True,
                                            id=id("spheroid_toggle")
                                        )
                                    )
                                ],
                                class_name="mb-3"
                            )
                        ],
                        class_name="col-6-checkbox"
                    ),
                ]
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "Z axis"
                                    ),
                                    dbc.Select(
                                        options=[
                                            {
                                                "label": "Bond",
                                                "value": "bond"
                                            },
                                            {
                                                "label": "Vector",
                                                "value": "vector"
                                            }
                                        ],
                                        value="bond",
                                        id=id("align_type"),
                                        style={"textAlign": "center"}
                                    )
                                ],
                                class_name="mb-3"
                            )
                        ],
                        class_name="col-6"
                    ),
                    dbc.Col(
                        id=id("bond_select_wrapper"),
                        children=dbc.InputGroup(
                            [
                                dbc.Select(
                                    options=default_bonds,
                                    value="0,6",
                                    id=id("bond_select"),
                                    style={
                                        "textAlign": "center"
                                    }
                                )
                            ],
                            className="mb-3"
                        ),
                    ),
                    dbc.Col(
                        id=id("vector_select_wrapper"),
                        children=[
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id=id('vector_x_select'),
                                        placeholder="x",
                                        type="number",
                                        value=0.0,
                                        style={
                                            "textAlign": "center"
                                        }
                                    ),
                                    dbc.Input(
                                        id=id('vector_y_select'),
                                        placeholder="y",
                                        type="number",
                                        value=0.0,
                                        style={
                                            "textAlign": "center"
                                        }
                                    ),
                                    dbc.Input(
                                        id=id('vector_z_select'),
                                        placeholder="z",
                                        type="number",
                                        value=1.0,
                                        style={
                                            "textAlign": "center"
                                        }
                                    )
                                ],
                                className="mb-3"
                            )
                        ],
                        style={"display": "none"}
                    )
                ]
            ),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Colour"
                            ),
                            dbc.Input(
                                id=id("spheroid_colour"),
                                type="color",
                                value="#6D1773",
                                style={
                                    "height": "40px"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Scale"
                            ),
                            dbc.Input(
                                id=id("spheroid_scale"),
                                type="number",
                                value=10,
                                style={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ])
            ])
        ]
    )
]

axis_options = [
    html.Div(
        children=[
            dbc.Row(
                dbc.Col(
                    html.H5(
                        style={"textAlign": "center"},
                        children="Axis",
                        className="mb-3"
                    )
                )
            ),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Axis"
                            ),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=True,
                                    id=id("axis_toggle")
                                )
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Colour"
                            ),
                            dbc.Input(
                                id=id("axis_colour"),
                                type="color",
                                value="#E13022",
                                style={
                                    "height": "40px"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Width"
                            ),
                            dbc.Input(
                                id=id("axis_width"),
                                type="number",
                                value=0.25,
                                style={
                                    "textAlign": "center"
                                },
                                min=0.
                            ),
                            dbc.InputGroupText(
                                "\u212B"
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Length"
                            ),
                            dbc.Input(
                                id=id("axis_length"),
                                type="number",
                                value=10,
                                style={
                                    "textAlign": "center"
                                }
                            ),
                            dbc.InputGroupText(
                                "\u212B"
                            )
                        ],
                        class_name="mb-3",
                    ),
                ])
            ])
        ]
    )
]

viewer_options = [html.Div(
    children=[
        dbc.Row(
            dbc.Col(
                html.H5(
                    style={"textAlign": "center"},
                    children="Viewer",
                    className="mb-3"
                )
            )
        ),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "X"
                        ),
                        dbc.Input(
                            id=id("view_x"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Y"
                        ),
                        dbc.Input(
                            id=id("view_y"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Z"
                        ),
                        dbc.Input(
                            id=id("view_z"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "Zoom"
                        ),
                        dbc.Input(
                            id=id("view_zoom"),
                            value="",
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "qX"
                        ),
                        dbc.Input(
                            id=id("view_qx"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "qY"
                        ),
                        dbc.Input(
                            id=id("view_qy"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "qZ"
                        ),
                        dbc.Input(
                            id=id("view_qz"),
                            value=0,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            "qW"
                        ),
                        dbc.Input(
                            id=id("view_qw"),
                            value=1,
                            type="number"
                        )
                    ],
                    className="mb-3"
                )
            )
        ]),
        dbc.Row([
            dbc.Col(
                [
                    dbc.Button(
                        "Download image",
                        color="primary",
                        className="me-1",
                        id=id("download_image_btn")
                    ),
                    html.Div(id=id('hidden-div1'), style={'display': 'none'})
                ],
                className="mb-3",
                style={"textAlign": "center"}
            ),
            dbc.Col(
                [
                    dbc.Button(
                        "Download 4f density .cube file",
                        color="primary",
                        className="me-1",
                        id=id("download_cube_btn")
                    ),
                    dcc.Download(id=id("download_cube_trigger")),
                    html.Div(id=id('hidden-div2'), style={'display': 'none'})
                ],
                className="mb-3",
                style={"textAlign": "center"}
            ),
            html.Div(id=id('hidden-div3'), style={'display': 'none'})
        ], className="align-items-center"),
    ]
)]

molecule_options = [
    html.Div(
        children=[
            dbc.Row(
                dbc.Col([
                    html.H5(
                        style={"textAlign": "center"},
                        children="Molecule",
                        className="mb-3"
                    )
                ])
            ),
            dbc.Row([
                dbc.Col(
                    [
                        dcc.Upload(
                            children=[
                                dbc.Button(
                                    "Select .xyz file",
                                    color="primary",
                                    className="me-1"
                                ),
                            ],
                            style={
                                'width': '100%',
                                'height': '100%',
                                'textAlign': 'center',
                            },
                            id=id("xyz_file"),
                            contents=default_xyz,
                        )
                    ],
                    style={"textAlign": "center"}
                ),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                id=id("centre_text"),
                                children="Centre"
                            ),
                            dbc.Tooltip(
                                children="Centre of spheroid",
                                target=id("centre_text"),
                                style={
                                    "textAlign": "center"
                                }
                            ),
                            dbc.Select(
                                options=[
                                    {
                                        "label": "C.O.M.",
                                        "value": "COM"
                                    },
                                    {
                                        "label": "Dy1",
                                        "value": "Dy1"
                                    }
                                ],
                                value="Dy1",
                                id=id('centre_select'),
                                style={
                                    "textAlign": "center",
                                    "verticalAlign": "middle",
                                    "horizontalAlign": "middle",
                                    "alignItems": "auto",
                                    "display": "inline"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Molecule"
                            ),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=True,
                                    id=id("molecule_toggle")
                                )
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Labels"
                            ),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=False,
                                    id=id("labels_toggle")
                                )
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
                dbc.Col([
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                "Style"
                            ),
                            dbc.Select(
                                options=[
                                    {
                                        "label": "Space-fill",
                                        "value": "sphere"
                                    },
                                    {
                                        "label": "Stick",
                                        "value": "stick"
                                    }
                                ],
                                value="stick",
                                id=id("molecule_style"),
                                style={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        class_name="mb-3",
                    ),
                ]),
            ])
        ]
    )
]

options = surface_options + molecule_options + spheroid_options
options += axis_options + viewer_options

# Layout of webpage
layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                dcc.Loading(
                                    children=[
                                        html.Div(
                                            id="density_mol_div",
                                            className="molecule_div"
                                        ),
                                        dcc.Store(id=id("atom_store")),
                                        dcc.Store(id=id("spheroid_store")),
                                        dcc.Store(id=id("axis_store")),
                                        dcc.Store(id=id("style_store")),
                                        dcc.Store(id=id("labels_store"))
                                    ]
                                ),
                            ],
                            className="col-6-viewer"
                        ),
                        html.Div(
                            options,
                            className="col-6-options",
                            style={
                                "overflow-y": "auto",
                                "overflow-x": "hidden"
                            }
                        )
                    ],
                    style={
                        "marginTop": "10px",
                        "width": "95vw",
                        "marginLeft": "2.5vw",
                        "marginRight": "2.5vw"
                    }
                )
            ],
            className="main_wrapper"
        ),
        utils.footer()
    ]
)

outputs = [
    Output(id("L_S_wrapper"), "style"),
]
inputs = [Input(id("n_value"), "value")]


@callback(outputs, inputs)
def toggle_l_s(n):
    """
    Toggles appearance of L and S entry fields based on number of
    electrons. L and S are only needed when n < 7
    """

    if n == 7:
        raise PreventUpdate
    elif n > 7:
        on_off = {"display": "none"}
    else:
        on_off = {}

    return [on_off]


outputs = [
    Output(id("atom_store"), "data"),
    Output(id("style_store"), "data"),
    Output(id("labels_store"), "data"),
    Output(id("axis_store"), "data"),
    Output(id("bond_select"), "options"),
    Output(id("centre_select"), "options"),
    Output(id("centre_select"), "value"),
    Output(id("bond_select"), "value"),
    Output(id("vector_select_wrapper"), "style"),
    Output(id("bond_select_wrapper"), "style"),

]

inputs = [
    Input(id("xyz_file"), "contents"),
    Input(id("molecule_toggle"), "value"),
    Input(id("molecule_style"), "value"),
    Input(id("axis_colour"), "value"),
    Input(id("axis_width"), "value"),
    Input(id("axis_length"), "value"),
    Input(id("bond_select"), "value"),
    [
        Input(id("vector_x_select"), "value"),
        Input(id("vector_y_select"), "value"),
        Input(id("vector_z_select"), "value")
    ],
    Input(id("centre_select"), "value"),
    Input(id("align_type"), "value")
]


@callback(outputs, inputs)
def update_app(xyz_file, mol_toggle, mol_style, axis_colour, axis_width,
               axis_length, bond_select, vector_select, centre_label,
               align_type):

    elements = [
        xyz_file, mol_toggle, mol_style, axis_colour, axis_width, axis_length,
        bond_select, vector_select, centre_label, align_type
    ]

    # Check for Nonetype
    if None in elements:
        raise PreventUpdate

    # Read labels and coordinates
    labels, coords = utils.parse_xyz_file(
        xyz_file
    )
    labels, labels_nn, coords = utils.process_labels_coords(labels, coords)

    # Check if structure has been changed, if so reset alignment vector
    # and centre label
    button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if id("xyz_file") in button_id:
        # Regenerate list of alignments to choose from
        align_choices = utils.create_bond_choices(labels, coords)
        bond_select_display = "{:d},{:d}".format(*align_choices[0]["value"])
        bond_select = bond_select_display
        centre_label = labels[0]
        centre_label_display = centre_label
    else:
        centre_label_display = no_update
        align_choices = no_update
        bond_select_display = no_update

    # Choose z-alignment vector from either bonds or raw vector input
    if align_type == "bond":
        if len(bond_select):
            bond_select = [int(ab) for ab in bond_select.split(",")]
            z_vec = coords[bond_select[1]] - coords[bond_select[0]]
        else:
            z_vec = [0., 0., 1.]
        bond_toggle = {}
        vector_toggle = {"display": "none"}
    elif align_type == "vector":
        print(vector_select)
        if None in vector_select:
            vector_select = [0., 0., 1.]
        z_vec = np.array([float(x) for x in vector_select])
        bond_toggle = {"display": "none"}
        vector_toggle = {}

    # Rotate molecular z axis onto specified z axis
    mol_z = [0., 1., 0.]
    if np.sum(np.abs(z_vec - mol_z)) != 0.:
        coords = set_z_alignment(coords, z_vec, mol_z)

    # Create list of atom labels for centering, and set center
    centre_choices, centre_index, _ = utils.define_centres(
        labels,
        coords,
        centre_label
    )

    # Shift coords to centre
    coords -= coords[centre_index]

    # Create list of bonding dictionaries
    if len(labels) > 1:
        adjacency = utils.get_adjacency(
            labels,
            coords,
            adjust_cutoff={"Dy": 1.6}
        )

    # Create molecule
    molecule = utils.make_js_molecule(
        coords, labels_nn, adjacency
    )
    # Style molecule
    mol_style_js = utils.make_js_molstyle(mol_style, labels_nn, 'atomic', 'm')

    # Add Numbered atom labels
    labels_js = utils.make_js_label(coords, labels, viewer_var="viewer")

    # Create axis cylinder
    axis = utils.make_js_cylinder(
        start_coords=[0, +axis_length/2., 0],
        end_coords=[0, -axis_length/2., 0],
        color=axis_colour,
        width=axis_width,
        viewer_var='viewer'
    )

    return molecule, mol_style_js, labels_js, axis, align_choices, \
        centre_choices, centre_label_display, bond_select_display, \
        vector_toggle, bond_toggle


outputs = [
    Output(id("spheroid_store"), "data"),
    Output(id("J_value"), "invalid"),
    Output(id("mJ_value"), "invalid"),
    Output(id("L_value"), "invalid"),
    Output(id("S_value"), "invalid"),
    Output(id("n_value"), "invalid"),
]

inputs = [
    Input(id("J_value"), "value"),
    Input(id("mJ_value"), "value"),
    Input(id("L_value"), "value"),
    Input(id("S_value"), "value"),
    Input(id("n_value"), "value"),
    Input(id("spheroid_scale"), "value")
]


@callback(outputs, inputs)
def update_spheroid(J, mJ, L, S, n, scale):
    invalidity = {
        "J": False,
        "mJ": False,
        "L": False,
        "S": False,
        "n": False
    }

    # Check for Nonetype
    if None in [J, mJ, L, S, n, scale]:
        spheroid = no_update

    # Less than half filled, ignore L and S (set to dummy values)
    if n > 7:
        L = 10
        S = 10

    for name, qn in zip(["J", "mJ", "L", "S"], [2*J, 2*mJ, L, 2*S]):
        if np.ceil(qn) != qn:
            spheroid = no_update
            invalidity[name] = True

    if n < 7:
        if J not in np.arange(np.abs(L-S), L + S + 1., 1.):
            spheroid = no_update
            invalidity["J"] = True
        elif L <= 0:
            spheroid = no_update
            invalidity["L"] = True
        elif S <= 0:
            spheroid = no_update
            invalidity["S"] = True

    if J <= 0:
        spheroid = no_update
        invalidity["J"] = True

    if mJ > J or mJ <= 0:
        spheroid = no_update
        invalidity["mJ"] = True

    # Check number of f-electrons
    if n < 1 or n > 13 or n == 7:
        spheroid = no_update
        invalidity["n"] = True

    if not any(invalidity.values()):
        spheroid = create_spheroid_iso(n, J, mJ, L, S, scale)

    return spheroid, *invalidity.values()


# Clientside callback for image download
clientside_callback(
    """
    function (dummy) {

        let canvas = document.getElementById("viewer_canvas");
        if (canvas == null){
            return;
        }
        var duri = canvas.toDataURL('image/png', 1)
        downloadURI(duri, "density.png");

        return ;
        }
    """, # noqa
    Output(id('hidden-div1'), "children"),
    [
        Input(id("download_image_btn"), "n_clicks"),
    ]
)


@callback(
    Output(id("download_cube_trigger"), "data"),
    [
        Input(id("download_cube_btn"), "n_clicks"),
        Input(id("spheroid_store"), "data")
    ],
    prevent_initial_call=True,
)
def func(n_clicks, data_str):
    if callback_context.triggered_id == id("spheroid_store"):
        return
    else:
        return dict(content=data_str, filename="spheroid.cube")


# Clientside callback for molecule viewer
clientside_callback(
    """
    function (atoms_spec, mol_style_js, spheroid, labels_js, axis, molecule_toggle, spheroid_toggle, labels_toggle, axis_toggle, spheroid_colour, x, y, z, zoom, qx, qy, qz, qw) {

        let element = document.getElementById("density_mol_div");

        while(element.firstChild){
            element.removeChild(element.firstChild);
        }

        let config = { backgroundOpacity: 0.};
        let viewer = $3Dmol.createViewer(element, config);

        viewer.getCanvas()["style"]="width: 100vw;"
        viewer.getCanvas()["id"]="viewer_canvas"

        var atoms = eval(atoms_spec)
        var m = viewer.addModel();
        if (molecule_toggle) {
            m.addAtoms(atoms);
            eval(mol_style_js);
        }
        if (spheroid_toggle) {
            var voldata = new $3Dmol.VolumeData(spheroid, "cube");
            viewer.addIsosurface(voldata, {isoval: 0.55, color: spheroid_colour, smoothness: 5});
        }

        if (labels_toggle) {
            eval(labels_js);
        }

        if (axis_toggle) {
            eval(axis);
        }
        
        viewer.render();

        if (document.getElementById("aniso_view_zoom").value == ""){
            viewer.zoomTo();
            zoom_level = viewer.getView()[3]
        }
        else {
            zoom_level = parseFloat(document.getElementById("aniso_view_zoom").value)
        }

        viewer.setView([
            parseFloat(document.getElementById("aniso_view_x").value),
            parseFloat(document.getElementById("aniso_view_y").value),
            parseFloat(document.getElementById("aniso_view_z").value),
            zoom_level,
            parseFloat(document.getElementById("aniso_view_qx").value),
            parseFloat(document.getElementById("aniso_view_qy").value),
            parseFloat(document.getElementById("aniso_view_qz").value),
            parseFloat(document.getElementById("aniso_view_qw").value)
        ])

        // Mouse movement 
        viewer.getCanvas().addEventListener("wheel", (event) => { updateViewText("aniso", viewer) }, false)
        viewer.getCanvas().addEventListener("mouseup", (event) => { updateViewText("aniso", viewer) }, false)
        viewer.getCanvas().addEventListener("touchend", (event) => { updateViewText("aniso", viewer) }, false)

        console.log(zoom_level)
        return zoom_level
        }
    """, # noqa
    Output(id("view_zoom"), "value"),
    [
        Input(id("atom_store"), "data"),
        Input(id("style_store"), "data"),
        Input(id("spheroid_store"), "data"),
        Input(id("labels_store"), "data"),
        Input(id("axis_store"), "data"),
        Input(id("molecule_toggle"), "value"),
        Input(id("spheroid_toggle"), "value"),
        Input(id("labels_toggle"), "value"),
        Input(id("axis_toggle"), "value"),
        Input(id("spheroid_colour"), "value"),
        Input(id("view_x"), "value"),
        Input(id("view_y"), "value"),
        Input(id("view_z"), "value"),
        Input(id("view_zoom"), "value"),
        Input(id("view_qx"), "value"),
        Input(id("view_qy"), "value"),
        Input(id("view_qz"), "value"),
        Input(id("view_qw"), "value")
    ]
)


def create_spheroid_iso(n, J, mJ, L, S, scale):
    """
    Creates an object which specifies an isosurface corresponding to the
    Sievers charge density surface for a given electronic configuration of
    the f shell.

    Parameters
    ----------
    n : int
        Number of f electrons
    J : float
        J quantum number
    mJ : float
        mJ quantum number
    L : float
        L quantum number
    S : float
        S quantum number
    scale : float
        Arbitrary scale factor for isosurface size

    Returns
    -------
    str
        contents of isosurface cube file as string
    """

    if n > 7:
        comment = "{:d} f electrons, J={:.1f}, mJ={:.1f}".format(n, J, mJ)
    else:
        comment = "{:d} f electrons, J={:.1f}, mJ={:.1f}, L={:d}, S={:.1f}".format( # noqa
            n, J, mJ, L, S
        )

    a_vals = sievers.compute_a_vals(n, J, mJ, L, S)
    cube_str = sievers.compute_isosurface(
        *a_vals,
        40,
        40,
        40,
        scale=scale,
        comment=comment
    )

    return cube_str


def set_z_alignment(coords, new_z, old_z):
    """
    Calculates rotation matrix which rotates old_z onto new_z, then applies
    this rotation to coordinates

    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates as (n_atoms,3) array
    new_z : np.ndarray
        Vector of new z axis in old_z frame
    old_z : np.ndarray
        Vector of old z axis in old_z frame

    Returns
    -------
    np.ndarray
        Coordinates after rotation
    """

    new_z /= la.norm(new_z)
    new_z = np.array(new_z)

    x = np.cross(new_z, old_z)/la.norm(np.cross(new_z, old_z))

    A = np.array(
        [
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ]
    )
    theta = np.arccos(np.dot(new_z, old_z)/(la.norm(new_z)*la.norm(old_z)))

    # Rodriguez rotation
    R = np.eye(3, 3) + np.sin(theta)*A + (1-np.cos(theta))*(A @ A)

    coords = (R @ coords.T).T

    return coords
