#! /usr/bin/env python3

from dash import html, dcc, callback_context, register_page, callback, \
    clientside_callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import io

from .core import orbitals as orbs
from .core import utils


PAGE_NAME = 'Orbitals'
PAGE_PATH = '/orbitals'
PAGE_IMAGE = 'assets/Orbitals.png'
PAGE_DESCRIPTION = "Interactive atomic orbitals and radial wavefunctions"

register_page(
    __name__,
    order=1,
    path=PAGE_PATH,
    name=PAGE_NAME,
    title=PAGE_NAME,
    image=PAGE_IMAGE,
    description=PAGE_DESCRIPTION
)

id = utils.dash_id("orb")


main_plot = html.Div(
    id=id("main_visual_wrapper"),
    children=[
        html.Div(
            id=id("2d_visual_wrapper"),
            children=[
                dcc.Graph(
                    id=id("2d_plot"),
                    className='plot_area',
                    mathjax=True
                )
            ]
        ),
        html.Div(
            id=id("3d_visual_wrapper"),
            children=[
                dcc.Loading(
                    children=[
                        html.Div(
                            id="orb_mol_div",
                            className="molecule_div"
                        ),
                        dcc.Store(id=id("orbital_store"), data=""),
                        dcc.Store(id=id("isoval_store"), data=0),
                    ]
                )
            ]
        ),
    ]
)

orb_name_2d = dcc.Dropdown(
    id=id("orb_name_2d"),
    style={
        "textAlign": "left"
    },
    options=[
        {"label": "1s", "value": "1s"},
        {"label": "2s", "value": "2s"},
        {"label": "3s", "value": "3s"},
        {"label": "4s", "value": "4s"},
        {"label": "5s", "value": "5s"},
        {"label": "6s", "value": "6s"},
        {"label": "7s", "value": "7s"},
        {"label": "2p", "value": "2p"},
        {"label": "3p", "value": "3p"},
        {"label": "4p", "value": "4p"},
        {"label": "5p", "value": "5p"},
        {"label": "6p", "value": "6p"},
        {"label": "7p", "value": "7p"},
        {"label": "3d", "value": "3d"},
        {"label": "4d", "value": "4d"},
        {"label": "5d", "value": "5d"},
        {"label": "6d", "value": "6d"},
        {"label": "7d", "value": "7d"},
        {"label": "4f", "value": "4f"},
        {"label": "5f", "value": "5f"},
        {"label": "6f", "value": "6f"},
        {"label": "7f", "value": "7f"},
    ],
    value=["1s", "2p", "3d", "4f"],
    multi=True,  # browser autocomplete needs to be killed here, when they implement it # noqa
    placeholder="Orbital..."
)

orb_name_3d = dbc.Select(
    id=id("orb_name_3d"),
    style={
        "textAlign": "left"
    },
    options=[
        {"label": "1s", "value": "1s"},
        {"label": "2s", "value": "2s"},
        {"label": "3s", "value": "3s"},
        {"label": "4s", "value": "4s"},
        {"label": "5s", "value": "5s"},
        {"label": "6s", "value": "6s"},
        {"label": "2p", "value": "2p"},
        {"label": "3p", "value": "3p"},
        {"label": "4p", "value": "4p"},
        {"label": "5p", "value": "5p"},
        {"label": "6p", "value": "6p"},
        {"label": "3dz²", "value": "3dz2"},
        {"label": "4dz²", "value": "4dz2"},
        {"label": "5dz²", "value": "5dz2"},
        {"label": "6dz²", "value": "6dz2"},
        {"label": "3dxy", "value": "3dxy"},
        {"label": "4dxy", "value": "4dxy"},
        {"label": "5dxy", "value": "5dxy"},
        {"label": "6dxy", "value": "6dxy"},
        {"label": "4fz³", "value": "4fz3"},
        {"label": "5fz³", "value": "5fz3"},
        {"label": "6fz³", "value": "6fz3"},
        {"label": "4fxyz", "value": "4fxyz"},
        {"label": "5fxyz", "value": "5fxyz"},
        {"label": "6fxyz", "value": "6fxyz"},
        {"label": "4fyz²", "value": "4fyz2"},
        {"label": "5fyz²", "value": "5fyz2"},
        {"label": "6fyz²", "value": "6fyz2"},
    ],
    value="",
    placeholder="Select an orbital"
)


orb_select = [
    dbc.Row([
        dbc.Col(
            html.H4(
                style={
                    "textAlign": "center",
                    },
                children="Orbital"
            )
        ),
        dbc.Col(
            html.H4(
                style={
                    "textAlign": "center",
                    },
                children="Function"
            )
        )
    ]),
    dbc.Row([
        dbc.Col(
            children=[
                html.Div(
                    id=id("orb_name_2d_container"),
                    children=orb_name_2d,
                    style={}
                ),
                html.Div(
                    id=id("orb_name_3d_container"),
                    children=orb_name_3d,
                    style={"display": None}
                )
            ]
        ),
        dbc.Col(
            children=[
                dbc.Select(
                    id=id("function_type"),
                    style={
                        "textAlign": "center",
                        "display": "block"
                    },
                    options=[
                        {
                         "label": "Radial Distribution Function",
                         "value": "RDF"
                        },
                        {
                         "label": "Radial Wave Function",
                         "value": "RWF"
                        },
                        {
                         "label": "3D Surface",
                         "value": "3DWF"
                        }
                    ],
                    value="RDF",
                )
            ]
        )
    ])
]

save_options = [
    html.Div(
        id=id("save_container"),
        children=[
            dbc.Row([
                dbc.Col([
                    html.H4(
                        style={
                            "textAlign": "center",
                        },
                        children="Save Options",
                        id=id("save_options_header")
                    ),
                    dbc.Tooltip(
                        children="Use the camera button in the top right of \
                                the plot to save an image",
                        target=id("save_options_header"),
                        style={
                            "textAlign": "center",
                        },
                    )
                ])
            ]),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Output height"),
                                dbc.Input(
                                    id=id("save_height_in"),
                                    placeholder=500,
                                    type="number",
                                    value=500,
                                    style={
                                        "textAlign": "center",
                                        "verticalAlign": "middle",
                                        "horizontalAlign": "middle"
                                    }
                                ),
                                dbc.InputGroupText("px"),
                            ],
                            class_name="mb-3",
                        ),
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Output width"),
                                dbc.Input(
                                    id=id("save_width_in"),
                                    placeholder=500,
                                    type="number",
                                    value=500,
                                    style={
                                        "textAlign": "center",
                                        "verticalAlign": "middle",
                                        "horizontalAlign": "middle"
                                    }
                                ),
                                dbc.InputGroupText("px"),
                            ],
                            class_name="mb-3",
                        ),
                    )
                ]
            ),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        id=id("download_link_box"),
                        children=[
                            dbc.Button(
                                "Download Data",
                                id=id("download_data"),
                                style={
                                    'boxShadow': 'none',
                                    'textalign': 'top'
                                }
                            ),
                            dcc.Download(id=id("download_data_trigger")),
                            dcc.Store(id=id("download_store"), data="")
                        ]
                    ),
                    style={"textAlign": "center"}
                ),
                dbc.Col(
                    [
                        dbc.InputGroup([
                            dbc.InputGroupText("Image format"),
                            dbc.Select(
                                id=id("save_format"),
                                style={
                                    "textAlign": "center",
                                    "horizontalAlign": "center",
                                    "display": "inline"
                                },
                                options=[
                                    {
                                        "label": "svg",
                                        "value": "svg",
                                    },
                                    {
                                        "label": "png",
                                        "value": "png",
                                    },
                                    {
                                        "label": "jpeg",
                                        "value": "jpeg",
                                    }
                                ],
                                value="svg"
                            )
                        ])
                    ]
                )
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
            ),
            className="mb-3"
        ),
        dbc.Row([
            dbc.Col(
                [
                    dbc.Button(
                        "Download image",
                        color="primary",
                        className="me-1",
                        id=id("download_button")
                    ),
                    html.Div(id=id('hidden-div'), style={'display': 'none'})
                ],
                className="mb-3",
                style={"textAlign": "center"}
            )
        ], className="align-items-center"),
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
        ])
    ],
    id=id("orb_3d_viewer_options"),
    style={
        "display": "none"
    }
)
]

orb_customise_2d = [
    dbc.Row(
        dbc.Col(
            html.H4(
                style={"textAlign": "center"},
                children="Plot Options"
            )
        )
    ),
    dbc.Row(
        children=[
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Lower x limit"),
                        dbc.Input(
                            id=id("lower_x_in"),
                            placeholder=0,
                            type="number",
                            min=-10,
                            max=100,
                            value=0,
                            style={
                                "textAlign": "center",
                                "verticalAlign": "middle",
                                "horizontalAlign": "middle"
                            }
                        )
                    ],
                    class_name="mb-3",
                ),
            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Upper x limit"),
                        dbc.Input(
                            id=id("upper_x_in"),
                            placeholder=40,
                            type="number",
                            min=0,
                            max=100,
                            value=40,
                            style={
                                "textAlign": "center",
                                "verticalAlign": "middle",
                                "horizontalAlign": "middle"
                            }
                        )
                    ],
                    class_name="mb-3",
                ),
            )
        ]
    ),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Gridlines"),
                    dbc.InputGroupText(
                        dbc.Checkbox(value=False, id=id("gridlines"))
                    )
                ],
                class_name="mb-3",
                style={"textAlign": "center"}
            ),
            style={"textAlign": "center"}
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Colour Palette"),
                    dbc.Select(
                        id=id("colours_2d"),
                        options=[
                            {
                                "label": "Standard",
                                "value": "normal"
                            },
                            {
                                "label": "Tol",
                                "value": "tol"
                            },
                            {
                                "label": "Wong",
                                "value": "wong"
                            }
                        ],
                        value="normal",
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle",
                            "alignItems": "auto",
                            "display": "inline"
                        }
                    )
                ]
            )
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Linewidth"),
                    dbc.Input(
                        id=id("linewidth"),
                        placeholder=5,
                        type="number",
                        min=1,
                        max=10,
                        value=5,
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        }
                    )
                ],
                class_name="mb-3",
            ),
            class_name="mb-3",
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Text size"),
                    dbc.Input(
                        id=id("text_size"),
                        placeholder=19,
                        type="number",
                        min=15,
                        max=25,
                        value=19,
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        }
                    )
                ],
                class_name="mb-3",
            ),
            class_name="mb-3",
        ),
    ])
]

orb_customise_3d = [
    dbc.Row([
        dbc.Col([
            html.H4(
                style={
                    "textAlign": "center",
                },
                children="Plot Options"
            )
        ])
    ]),
    dbc.Row(
        children=[
            dbc.Col([
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Colours"),
                        dbc.Select(
                            id=id("colours_3d_a"),
                            options=[
                                {
                                    "label": "Purple",
                                    "value": "purple"
                                },
                                {
                                    "label": "Blue",
                                    "value": "blue"
                                },
                                {
                                    "label": "Orange",
                                    "value": "orange"
                                },
                                {
                                    "label": "Yellow",
                                    "value": "yellow"
                                },
                                {
                                    "label": "Red",
                                    "value": "red"
                                },
                                {
                                    "label": "Green",
                                    "value": "green"
                                }
                            ],
                            style={"textAlign": "center"},
                            value="yellow",
                        ),
                        dbc.Select(
                            id=id("colours_3d_b"),
                            options=[
                                {
                                    "label": "Purple",
                                    "value": "purple"
                                },
                                {
                                    "label": "Blue",
                                    "value": "blue"
                                },
                                {
                                    "label": "Orange",
                                    "value": "orange"
                                },
                                {
                                    "label": "Yellow",
                                    "value": "yellow"
                                },
                                {
                                    "label": "Red",
                                    "value": "red"
                                },
                                {
                                    "label": "Green",
                                    "value": "green"
                                }
                            ],
                            style={},
                            value="purple",
                        )
                    ],
                    class_name="mb-2"
                ),
            ])
        ]),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Cutaway"),
                    dbc.Select(
                        id=id("cutaway_in"),
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        },
                        options=[
                            {
                                "label": "None",
                                "value": 1.
                            },
                            {
                                "label": "1/2",
                                "value": 0.5
                            }
                        ],
                        value=1.
                    )
                ],
                class_name="mb-2"
            )
        ]),
        dbc.Col([
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Wireframe"),
                    dbc.InputGroupText(
                        dbc.Checkbox(value=False, id=id("wireframe"))
                    )
                ],
                class_name="mb-2"
            )
        ])
    ])
]

orb_customise = [
    html.Div(
        id=id("orb_customise_2d"),
        children=orb_customise_2d,
        style={},
        className="pt-5",
    ),
    html.Div(
        id=id("orb_customise_3d"),
        children=orb_customise_3d,
        style={"display": "none"},
        className="pt-5",
    )]


orb_options = orb_select + orb_customise + save_options
orb_options += viewer_options

# Layout of webpage
layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        dbc.Row([
                            dbc.Col(
                                children=[main_plot],
                                className="col-6"
                            ),
                            dbc.Col(
                                children=orb_options,
                                className="col-6"
                            )
                        ])
                    ],
                    style={
                        "marginTop": "10px",
                        "width": "95vw",
                        "marginLeft": "2.5vw",
                        "marginRight": "2.5vw"
                    }
                ),
            ],
            className="main_wrapper"
            ),
        utils.footer()
    ]
)


# Callback for download 2d data button
@callback(
    Output(id("download_data_trigger"), "data"),
    [
        Input(id("download_data"), "n_clicks"),
        Input(id("download_store"), "data")
    ],
    prevent_initial_call=True,
)
def func(n_clicks, data_str):
    if callback_context.triggered_id == id("download_store"):
        return
    else:
        return dict(content=data_str, filename="waveplot_orbital_data.dat")


outputs = [
    Output(id("2d_visual_wrapper"), "style"),
    Output(id("3d_visual_wrapper"), "style"),
    Output(id("orb_customise_2d"), "style"),
    Output(id("orb_customise_3d"), "style"),
    Output(id("orb_name_2d_container"), "style"),
    Output(id("orb_name_3d_container"), "style"),
    Output(id("orb_3d_viewer_options"), "style"),
    Output(id("download_link_box"), "style"),
    Output(id("save_container"), "style")
]


@callback(outputs, Input(id("function_type"), "value"))
def update_options(wf_type):
    """
    Callback to update layout of page depending on which type
    of orbital function is requested
    """

    on = {}
    off = {"display": "none"}

    if "3" in wf_type:
        displays = [off, on, off, on, off, on, on, off, off]
    else:
        displays = [on, off, on, off, on, off, off, on, on]

    return displays


outputs = [
    Output(id("2d_plot"), "figure"),
    Output(id("2d_plot"), "config"),
    Output(id("download_store"), "data")
]

inputs = [
    Input(id("orb_name_2d"), "value"),
    Input(id("function_type"), "value"),
    Input(id("linewidth"), "value"),
    Input(id("text_size"), "value"),
    Input(id("gridlines"), "value"),
    Input(id("upper_x_in"), "value"),
    Input(id("lower_x_in"), "value"),
    Input(id("save_format"), "value"),
    Input(id("save_height_in"), "value"),
    Input(id("save_width_in"), "value"),
    Input(id("colours_2d"), "value"),
]


@callback(outputs, inputs)
def update_2d_plot(orbital_names, wf_type, linewidth, text_size,
                   gridlines, x_up, x_low, save_format, save_height,
                   save_width, colours_2d):
    """
    Updates the 2d plot, given the current state of the UI
    All inputs correspond (in the same order) to the list
    of Inputs given above

    Parameters
    ----------
    orbital_names : str
        names of orbitals as strings
    wf_type : str {RWF, RDF, 3DWF}
        Type of wavefunction
    linewidth : float
        linewidth for 2d plot
    text_size : float
        Axis label text sizes
    gridlines : list
        yes or no (str) to gridlines on either axis
    x_up : float
        upper x limit for 2d plot
    x_low : float
        lower x limit for 2d plot
    save_format : str
        save format for plot
    save_height : float
        height of saved image
    save_width : float
        width of saved image
    colour_2d : str
        colour style for 2d plot

    Returns
    -------
    dict
        item for list of go.XX objects
    dict
        contains settings for 2d plot modebar
    str
        output file containing all data as string with correct encoding
        for download
    """

    # Stop update if 3d wavefunction requested
    if "3" in wf_type:
        raise PreventUpdate

    # Calculate RWF data
    full_data, fig = gen_2d_orbital_data(
        orbital_names,
        x_up,
        x_low,
        wf_type,
        linewidth,
        colours_2d,
        gridlines,
        text_size
    )

    output_str = make_output_file(wf_type, full_data, orbital_names)

    # Set modebar icons
    if len(orbital_names) == 0:
        modebar = {"displayModeBar": False}
    else:
        modebar = set_2d_modebar(
            save_format,
            save_height,
            save_width,
            wf_type
        )

    return fig, modebar, output_str


outputs = [
    Output(id("orbital_store"), "data"),
    Output(id("isoval_store"), "data")
]

inputs = [
    Input(id("orb_name_3d"), "value"),
    Input(id("cutaway_in"), "value")
]


@callback(outputs, inputs)
def update_3d_store(orbital_name, cutaway):
    """
    Updates the 3d orbital data store, which contains cube file for javascript
    orbital visualisation, and the isovalue store which contains the
    isovalue used in the j isosurface

    Parameters
    ----------
    orbital_name : str
        name of orbital as string
    cutaway : string
        controls 3d slicing of orbitals

    Returns
    -------

    """

    cutaway = float(cutaway)

    if cutaway is None and orbital_name == [""]:
        raise PreventUpdate

    if not len(orbital_name):
        raise PreventUpdate

    cube, isoval = gen_3d_orb_cube(
        orbital_name,
        cutaway
    )

    return cube, isoval


# Clientside callback for javascript molecule viewer
clientside_callback(
    """
    function (dummy) {

        let canvas = document.getElementById("viewer_canvas");
        if (canvas == null){
            return;
        }
        var duri = canvas.toDataURL('image/png', 1)
        downloadURI(duri, "orbital.png");

        return ;
        }
    """, # noqa
    Output(id('hidden-div'), "children"),
    [
        Input(id("download_button"), "n_clicks"),
    ]
)


# Clientside callback for javascript molecule viewer used to show 3d orbitals
clientside_callback(
    """
    function (orbital, orb_iso, colour_a, colour_b, wire_toggle, orb_name_3d, x, y, z, zoom, qx, qy, qz, qw) {

        let element = document.getElementById("orb_mol_div");

        while(element.firstChild){
            element.removeChild(element.firstChild);
        }

        let config = { backgroundOpacity: 0.};
        let viewer = $3Dmol.createViewer(element, config);

        viewer.getCanvas()["style"]="width: 100vw;"
        viewer.getCanvas()["id"]="viewer_canvas"

        var m = viewer.addModel();

        var voldata = new $3Dmol.VolumeData(orbital, "cube");
        viewer.addIsosurface(
            voldata,
            {
                isoval: orb_iso,
                color: colour_a,
                wireframe: wire_toggle,
                smoothness: 6
            }
        );
        viewer.addIsosurface(
            voldata,
            {
                isoval: -orb_iso,
                color: colour_b,
                wireframe: wire_toggle,
                smoothness: 6
            }
        );

        viewer.render();
        viewer.center();

        if (orb_iso !== 0) {
            if (document.getElementById("orb_view_zoom").value == ""){
                viewer.zoomTo();
                zoom_level = viewer.getView()[3]
            }
            else {
                zoom_level = parseFloat(document.getElementById("orb_view_zoom").value)
            }

            console.log(parseFloat(document.getElementById("orb_view_x").value))
            viewer.setView([
                parseFloat(document.getElementById("orb_view_x").value),
                parseFloat(document.getElementById("orb_view_y").value),
                parseFloat(document.getElementById("orb_view_z").value),
                zoom_level,
                parseFloat(document.getElementById("orb_view_qx").value),
                parseFloat(document.getElementById("orb_view_qy").value),
                parseFloat(document.getElementById("orb_view_qz").value),
                parseFloat(document.getElementById("orb_view_qw").value)
            ])
            viewer.getCanvas().addEventListener("wheel", (event) => { updateViewText("orb", viewer) }, false)
            viewer.getCanvas().addEventListener("mouseup", (event) => { updateViewText("orb", viewer) }, false)   
            viewer.getCanvas().addEventListener("touchend", (event) => { updateViewText("orb", viewer) }, false)   
            return zoom_level;
        }
        else {
            return ["", undefined];
        }

        }
    """,  # noqa
    [
        Output(id("view_zoom"), "value"),
    ],
    [
        Input(id("orbital_store"), "data"),
        Input(id("isoval_store"), "data"),
        Input(id("colours_3d_a"), "value"),
        Input(id("colours_3d_b"), "value"),
        Input(id("wireframe"), "value"),
        Input(id("orb_name_3d"), "value"),
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


def make_output_file(wf_type, full_data, orbital_names):
    """
    Create output file as string
    """
    data_header = "{}".format(wf_type)
    data_header += " Data generated using waveplot.com, an app by"
    data_header += " Jon Kragskow \n x (a0),  "

    for orb in orbital_names:
        data_header += "{}, ".format(orb)
    data_str = io.StringIO()
    np.savetxt(data_str, np.transpose(full_data), header=data_header)
    output_str = data_str.getvalue()

    return output_str


def gen_2d_orbital_data(orbital_names, x_up, x_low, wf_type, linewidth,
                        colours_2d, gridlines, text_size):
    """
    Creates list of plotly go.Scatter objects for plot, and array of all
    wavefunction data for 2D orbital plots of RWF and RDF

    Parameters
    ----------
    orbital_names : list
        orbital names as list of strings formatted as e.g. ['1s', '2p', '3d']
    x_up : float
        Maximum distance to calculate RDF/RWF at in units of bohr radius
    x_low : float
        Minimum distance to calculate RDF/RWF at in units of bohr radius
    wf_type : str {'RWF', 'RDF'}
        Type of wavefunction to plot
    linewidth : float
        Linewidth to use for all plots
    colours_2d : str {'tol', 'wong', 'normal'}
        Primary colour palette to use for plots as defined in `utils.py`

    Returns
    -------
    list
        All wavefunction data as list, first row is distance in bohr radii,
        all other rows are wavefunction data. Rows are ordered to match the
        input `orbital_names` list of strings
    list
        Plotly go.Scatter objects, one per `orbital`
    """
    # Nothing to plot - exit
    if len(orbital_names) == 0 or orbital_names is None:
        raise PreventUpdate

    # Set limits to default if needed
    if x_low is None or x_low < 0:
        x_low = 0
    if x_up is None:
        x_up = x_low + 100

    if text_size is None or text_size <= 3:
        text_size = 19

    # Set axis layout
    layout = set_2d_layout(
        wf_type,
        text_size,
        gridlines,
        x_up,
        x_low
    )

    # Create data
    full_data, data = create_2d_traces(
        orbital_names,
        x_up,
        x_low,
        wf_type,
        linewidth,
        colours_2d
    )

    output = {
        "data": data,
        "layout": layout
    }

    return full_data, output


def create_2d_traces(orbital_names, x_up, x_low, wf_type, linewidth,
                     colours_2d):
    """
    Creates list of plotly go.Scatter objects for plot, and array of all
    wavefunction data for 2D orbital plots of RWF and RDF

    Parameters
    ----------
    orbital_names : list
        orbital names as list of strings formatted as e.g. ['1s', '2p', '3d']
    x_up : float
        Maximum distance to calculate RDF/RWF at in units of bohr radius
    x_low : float
        Minimum distance to calculate RDF/RWF at in units of bohr radius
    wf_type : str {"RDF", "RWF"}
        Type of wavefunction to plot
    linewidth : float
        Linewidth to use for all plots
    colours_2d : str {'tol', 'wong', 'normal'}
        Primary colour palette to use for plots as defined in `utils.py`

    Returns
    -------
    list
        All wavefunction data as list, first row is distance in bohr radii,
        all other rows are wavefunction data. Rows are ordered to match the
        input `orbital_names` list of strings
    list
        Plotly go.Scatter objects, one per `orbital`
    """

    # Create colour list in correct order
    # i.e. selected colour is first
    if colours_2d == "tol":
        cols = utils.tol_cols + utils.wong_cols + utils.def_cols
    elif colours_2d == "wong":
        cols = utils.wong_cols + utils.def_cols + utils.tol_cols
    else:
        cols = utils.def_cols + utils.tol_cols + utils.wong_cols

    # List for individual traces
    traces = []
    full_data = []
    x = np.linspace(x_low, x_up, 1000)
    full_data.append(x)

    # Dictionary of orbital functions
    orb_funcs = {
        "s": orbs.s_2d,
        "p": orbs.p_2d,
        "d": orbs.d_2d,
        "f": orbs.f_2d,
    }

    # Create trace for each RWF or RDF
    for it, orbital in enumerate(orbital_names):
        # Get orbital n value and name
        n_qn = int(orbital[0])
        l_symb = orbital[1]

        # Calculate RWF or RDF data
        func_val = orb_funcs[l_symb](n_qn, x, wf_type)

        # Add to list of all data for output file
        full_data.append(func_val)

        # Create plotly trace
        traces.append(
            go.Scatter(
                x=x,
                y=func_val,
                line=dict(width=linewidth),
                name=orbital,
                hoverinfo="none",
                marker={"color": cols[it]}
            )
        )

    return full_data, traces


def gen_3d_orb_cube(orbital_name, cutaway):
    """
    Generates cube file containing orbital data

    Parameters
    ----------
    orbital_name : str
        name of orbital
    cutaway : float
        amount of cutaway

    Returns
    -------
    str
        cube file for selected orbital as string
    """

    n = int(orbital_name[0])
    name = orbital_name[1:]

    # Get orbital n value and name
    orb_func_dict = {
        "s": orbs.s_3d,
        "p": orbs.p_3d,
        "dxy": orbs.dxy_3d,
        "dz2": orbs.dz_3d,
        "fxyz": orbs.fxyz_3d,
        "fyz2": orbs.fyz2_3d,
        "fz3": orbs.fz_3d
    }

    n_points, wav, _, lower, isoval, step = orb_func_dict[name](
        n, cutaway=cutaway
    )

    step = np.abs(step)

    # Create Gaussian Cube file as string for current orbital
    cube = ""

    cube += "Comment line\n"
    cube += "Comment line\n"
    cube += "1     {:.6f} {:.6f} {:.6f}\n".format(lower, lower, lower)
    cube += "{:d}   {:.6f}    0.000000    0.000000\n".format(n_points, step)
    cube += "{:d}   0.000000    {:.6f}    0.000000\n".format(n_points, step)
    cube += "{:d}   0.000000    0.000000    {:.6f}\n".format(n_points, step)
    cube += " 1   0.000000    0.000000   0.000000  0.000000\n"

    a = 0

    for xit in range(n_points):
        for yit in range(n_points):
            for zit in range(n_points):
                a += 1
                cube += "{:.5e} ".format(wav[xit, yit, zit])
                if a == 6:
                    cube += "\n"
                    a = 0
            cube += "\n"
            a = 0

    return cube, isoval


def set_2d_layout(wf_type, text_size, gridlines, x_up, x_low):
    """
    creates go.Layout object for 2d plots

    Parameters
    ----------
    orbital_name : str
        name of orbital
    cutaway : float
        amount of cutaway

    Returns
    -------
    go.Layout
        plotly graph objects Layout object for current plot
    """

    y_labels = {
        "RDF": r"$4 \pi r^2 R(r)^2$",
        "RWF": r"$R(r)$"
    }

    layout = go.Layout(
                xaxis={
                    "autorange": True,
                    "showgrid": gridlines,
                    "zeroline": False,
                    "showline": True,
                    "range": [x_low, x_up],
                    "title": {
                        "text": r"$ r \ (a_0)$",
                        "font": {"size": text_size, "color": "black"}
                    },
                    "ticks": "outside",
                    "tickfont": {"size": text_size, "color": "black"},
                    "showticklabels": True
                },
                yaxis={
                    "autorange": True,
                    "showgrid": gridlines,
                    "zeroline": False,
                    "fixedrange": True,
                    "title": {
                        "text": y_labels[wf_type],
                        "font": {
                            "size": text_size,
                            "color": "black"
                        }
                    },
                    "title_standoff": 20,
                    "showline": True,
                    "ticks": "outside",
                    "tickfont": {"size": text_size, "color": "black"},
                    "showticklabels": True
                },
                legend={
                    "x": 0.8,
                    "y": 1,
                    "font": {
                        "size": text_size - 3
                    }
                },
                margin=dict(l=90, r=30, t=30, b=60),
    )

    return layout


def set_2d_modebar(save_format, save_height, save_width, wf_type):
    """
    Sets options of plotly's modebar for 2d plot. Removes all icons other than
    save image button.

    Parameters
    ----------
    save_format : str {"png", "svg", "jpeg"}
        Format of saved image
    save_height : int
        Height of saved image in pixels (px)
    save_widht : int
        Width of saved image in pixels (px)
    wf_type : str {"RDF", "RWF"}
        Type of wavefunction to plot

    Returns
    -------
    dict
        Options for plotly modebar
    """

    file_names = {
        "RDF": "radial_distribution_function",
        "RWF": "radial_wavefunction"
    }

    options = {
        "toImageButtonOptions": {
            "format": save_format,
            "filename": file_names[wf_type],
            "height": save_height,
            "width": save_width,
        },
        "modeBarButtonsToRemove": [
            "sendDataToCloud",
            "autoScale2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "toggleSpikelines",
            "zoom2d",
            "zoom3d",
            "pan3d",
            "pan2d",
            "select2d",
            "zoomIn2d",
            "zoomOut2d",
            "hovermode",
            "resetCameraLastSave3d",
            "hoverClosest3d",
            "hoverCompareCartesian",
            "resetViewMapbox",
            "orbitRotation",
            "tableRotation",
            "resetCameraDefault3d"
        ],
        "displaylogo": False,
        "displayModeBar": True,
    }

    return options
