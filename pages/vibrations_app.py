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

from dash import dcc, html, callback_context, register_page, callback, \
    clientside_callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import io
import urllib

from .core import vibrations
from .core import utils

PAGE_NAME = 'Vibrations'
PAGE_PATH = '/vibrations'
PAGE_IMAGE = 'assets/Vibrations.png'
PAGE_DESCRIPTION = "Interactive harmonic oscillator wavefunctions and energy levels" # noqa

register_page(
    __name__,
    order=2,
    path=PAGE_PATH,
    name=PAGE_NAME,
    title=PAGE_NAME,
    image=PAGE_IMAGE,
    description=PAGE_DESCRIPTION
)
id = utils.dash_id("vib")


def create_output_file(displacement, state_e, harmonic_e, harmonic_wf):
    """
    Creates output file for harmonic potential energies, state energies,
    and wavefunctions

    Parameters
    ----------
    displacement : np.ndarray
        Displacements used for classical oscillator in metres
    state_e : np.ndarray
        Harmonic state energies for quantum oscillator in wavenumbers
    harmonic_e : np.ndarray
        Harmonic energies for classical oscillator in wavenumbers
    harmonic_wf : np.ndarray
        Harmonic wavefunction for each state as a function of displacement
        2D array = [n_states, displacement.size]
    Returns
    -------
    str
        string containing output file
    """

    oc = io.StringIO()

    oc.write(
        "Vibrational wavefunction data calculated using waveplot.com\n" +
        "A tool by Jon Kragskow\n"
    )

    oc.write("\nState energies (cm-1)\n")
    for se in state_e:
        oc.write("{:.6f}\n".format(se))

    oc.write(
        "\nDisplacement (A), Harmonic potential (cm-1)\n"
    )
    for di, se in zip(displacement*10E10, harmonic_e):
        oc.write("{:.6f} {:.6f}\n".format(di, se))

    oc.write(
        "\nDisplacement (A), Harmonic Wavefunction for n=0, n=1, ...\n"
    )

    # transpose so rows are displacement
    harmonic_wf = harmonic_wf.T

    for di, row in zip(displacement*10E10, harmonic_wf):
        oc.write("{:.8f} ".format(di))
        for state_wf in row:
            oc.write("{:.8f} ".format(state_wf))
        oc.write("\n")

    return oc.getvalue()


main_plot = dcc.Graph(
    id=id("main_plot"),
    className='plot_area'
)

vib_options = [
    dbc.Row([
        dbc.Col([
            html.H4(
                style={
                    "textAlign": "center",
                },
                children="Parameters",
                id=id("parameters_header")
            ),
            dbc.Tooltip(
                children="Toggle buttons specify the two fixed parameters",
                target=id("parameters_header"),
                style={
                    "textAlign": "center",
                },
            )
        ])
    ]),
    dbc.Row(
        [
            dbc.Col(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(u"\u03BD"),
                            dbc.Input(
                                id=id("lin_wn"),
                                placeholder=2888,
                                type="number",
                                min=0.0001,
                                value=2888,
                                style={
                                    "textAlign": "center"
                                }
                            ),
                            dbc.InputGroupText(r"cm⁻¹"),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=True,
                                    id=id("lin_wn_fix")
                                )
                            )
                        ],
                        class_name="mb-3"
                    )
                ]
            ),
            dbc.Col(
                children=[
                    dbc.Row([
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("ω"),
                                dbc.Input(
                                    id=id("ang_wn"),
                                    placeholder=2888,
                                    type="number",
                                    min=0.0001,
                                    value=2888,
                                    style={
                                        "textAlign": "center"
                                    }
                                ),
                                dbc.InputGroupText(r"cm⁻¹"),
                                dbc.InputGroupText(
                                    dbc.Checkbox(
                                        value=False,
                                        id=id("ang_wn_fix")
                                    )
                                )
                            ],
                            class_name="mb-3"
                        )
                    ]),
                ]
            )
        ]
    ),
    dbc.Row(
        [
            dbc.Col(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("k"),
                            dbc.Input(
                                id=id("fc"),
                                placeholder=480,
                                value=480,
                                type="number",
                                min=0,
                                style={
                                    "textAlign": "center"
                                }
                            ),
                            dbc.InputGroupText(r"N m⁻¹"),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=True,
                                    id=id("fc_fix")
                                )
                            )
                        ],
                        class_name="mb-3"
                    )
                ]
            ),
            dbc.Col(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("μ"),
                            dbc.Input(
                                id=id("mu"),
                                placeholder=1,
                                value=1,
                                type="number",
                                min=0.00000001,
                                style={
                                    "textAlign": "center"
                                }
                            ),
                            dbc.InputGroupText(r"g mol⁻¹"),
                            dbc.InputGroupText(
                                dbc.Checkbox(
                                    value=False,
                                    id=id("mu_fix")
                                )
                            )
                        ],
                        class_name="mb-3"
                    )
                ]
            )
        ]
    )
]

plot_options = [
    dbc.Row(
        dbc.Col(
            html.H4(
                style={"textAlign": "center"},
                children="Plot Options"
            )
        )
    ),
    dbc.Row(
        [
            dbc.Col(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Max. n"),
                            dbc.Input(
                                id=id("max_n"),
                                placeholder=5,
                                value=5,
                                type="number",
                                min=0,
                                max=25,
                                style={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        class_name="mb-3"
                    )
                ]
            ),
            dbc.Col(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("WF scale"),
                            dbc.Input(
                                id=id("wf_scale"),
                                placeholder=1800,
                                type="number",
                                min=0.000000001,
                                value=1800,
                                style={
                                    "textAlign": "center"
                                }
                            )
                        ],
                        class_name="mb-3"
                    )
                ]
            )
        ]
    ),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("WF colour"),
                    dbc.Select(
                        id=id("wf_colour"),
                        options=[
                            {
                                "label": "Default",
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
                ],
                class_name="mb-2"
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("PE colour"),
                    dbc.Select(
                        id=id("pe_colour"),
                        options=[
                            {
                                "label": "Black",
                                "value": "black"
                            },
                            {
                                "label": "Blue",
                                "value": "blue"
                            },
                            {
                                "label": "Green",
                                "value": "green"
                            },
                            {
                                "label": "Red",
                                "value": "red"
                            }
                        ],
                        value="black",
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle",
                            "alignItems": "auto",
                            "display": "inline"
                        }
                    )
                ],
                class_name="mb-2"
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("State colour"),
                    dbc.Select(
                        id=id("state_colour"),
                        options=[
                            {
                                "label": "Black",
                                "value": "black"
                            },
                            {
                                "label": "Blue",
                                "value": "blue"
                            },
                            {
                                "label": "Green",
                                "value": "green"
                            },
                            {
                                "label": "Red",
                                "value": "red"
                            }
                        ],
                        value="black",
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle",
                            "alignItems": "auto",
                            "display": "inline"
                        }
                    )
                ],
                class_name="mb-2"
            )
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("WF linewidth"),
                    dbc.Input(
                        id=id("wf_linewidth"),
                        placeholder=3,
                        type="number",
                        min=1,
                        max=10,
                        value=3,
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        }
                    )
                ],
                class_name="mb-3",
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("PE linewidth"),
                    dbc.Input(
                        id=id("pe_linewidth"),
                        placeholder=2,
                        type="number",
                        min=1,
                        max=10,
                        value=2,
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        }
                    )
                ],
                class_name="mb-3",
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("State linewidth"),
                    dbc.Input(
                        id=id("state_linewidth"),
                        placeholder=2,
                        type="number",
                        min=1,
                        max=10,
                        value=2,
                        style={
                            "textAlign": "center",
                            "verticalAlign": "middle",
                            "horizontalAlign": "middle"
                        }
                    )
                ],
                class_name="mb-3",
            )
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Toggle WF"),
                    dbc.InputGroupText(
                        dbc.Checkbox(
                            value=True,
                            id=id("toggle_wf")
                        )
                    )
                ],
                class_name="mb-3",
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Toggle PE"),
                    dbc.InputGroupText(
                        dbc.Checkbox(
                            value=True,
                            id=id("toggle_pe")
                        )
                    )
                ],
                class_name="mb-3",
            )
        ),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Toggle states"),
                    dbc.InputGroupText(
                        dbc.Checkbox(
                            value=True,
                            id=id("toggle_states")
                        )
                    )
                ],
                class_name="mb-3",
            )
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Text size"),
                    dbc.Input(
                        id=id("text_size"),
                        placeholder=18,
                        type="number",
                        min=10,
                        max=25,
                        value=15,
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
        )
    ])
]

save_options = [
    dbc.Row([
        dbc.Col([
            html.H4(
                style={
                    "textAlign": "center",
                },
                children=""
            )
        ])
    ]),
    dbc.Row(
        [
            dbc.Col(
                html.Div(
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
                        dcc.Store(id=id("data_store"), data="")
                    ]
                ),
                style={"textAlign": "center"}
            ),
            dbc.Col(
                html.Div(
                    children=[
                        dbc.Button(
                            "Download Plot",
                            id=id("download_plot"),
                            style={
                                'boxShadow': 'none',
                                'textalign': 'top'
                            }
                        ),
                        html.Div(
                            id=id("hidden-div"), style={"display": "none"}
                        )
                    ]
                ),
                style={"textAlign": "center"}
            )
        ],
        class_name="mb-3"
    )
]

# Layout of webpage
layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        dbc.Row([
                            dbc.Col(
                                children=main_plot,
                                className="col-6"
                            ),
                            dbc.Col(
                                children=vib_options+plot_options+save_options,
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

# Image download callback
clientside_callback(
    """
    function (dummy) {

        // Get svg for plot and axis labels
        let plot = document.getElementById("vib_main_plot");
        if (plot == null || plot == undefined){
            return;
        }
        var plotData = plot.childNodes[1].firstElementChild.firstElementChild.childNodes[0];
        var plotContent = Array.from(plotData.childNodes);

        var labelData = plot.childNodes[1].firstElementChild.firstElementChild.childNodes[2];
        var labelContent = Array.from(labelData.childNodes);

        // merge svg together
        // createElementNS for svg
        var svgNS = "http://www.w3.org/2000/svg";  
        var mergedSvg = document.createElementNS(svgNS, 'svg');
        mergedSvg.setAttribute('id', 'merged');
        // keep the viewBox of the chart

        // adding the content of both svgs
        for (let i = 0; i < plotContent.length; i++) {
        mergedSvg.appendChild(plotContent[i]);
        }
        for (let i = 0; i < labelContent.length; i++) {
        mergedSvg.appendChild(labelContent[i]);
        }

        mergedSvg.setAttribute('width', plotData.getAttribute('width'));
        mergedSvg.setAttribute('height', plotData.getAttribute('height'));
        mergedSvg.setAttribute('style', plotData.getAttribute('style'));
        mergedSvg.setAttribute('xmlns', plotData.getAttribute('xmlns'));

        var svgBlob = new Blob([mergedSvg.outerHTML], {type:"image/svg+xml;charset=utf-8"});
        var svgUrl = URL.createObjectURL(svgBlob);
        var downloadLink = document.createElement("a");
        downloadLink.href = svgUrl;
        downloadLink.download = "vibrations.svg";
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);

        // Add the content back to their original homes
        for (let i = 0; i < plotContent.length; i++) {
        plotData.appendChild(plotContent[i]);
        }
        for (let i = 0; i < labelContent.length; i++) {
        labelData.appendChild(labelContent[i]);
        }

        return ;
        }
    """, # noqa
    Output(id("hidden-div"), "children"),
    [
        Input(id("download_plot"), "n_clicks")
    ],
    prevent_initial_call=True
)


@callback(
    Output(id("download_data_trigger"), "data"),
    [
        Input(id("download_data"), "n_clicks"),
        Input(id("data_store"), "data")
    ],
    prevent_initial_call=True,
)
def func(n_clicks, data_str):
    if callback_context.triggered_id == id("data_store"):
        return
    else:
        return dict(content=data_str, filename="waveplot_vibrational_data.dat")


@callback(
    [
        Output(id("main_plot"), "figure"),
        Output(id("main_plot"), "config"),
        Output(id("data_store"), "data"),
        Output(id("lin_wn"), "value"),
        Output(id("ang_wn"), "value"),
        Output(id("fc"), "value"),
        Output(id("mu"), "value"),
        Output(id("lin_wn"), "disabled"),
        Output(id("ang_wn"), "disabled"),
        Output(id("fc"), "disabled"),
        Output(id("mu"), "disabled"),
    ],
    [
        Input(id("lin_wn"), "value"),
        Input(id("ang_wn"), "value"),
        Input(id("fc"), "value"),
        Input(id("mu"), "value"),
        Input(id("max_n"), "value"),
        Input(id("lin_wn_fix"), "value"),
        Input(id("ang_wn_fix"), "value"),
        Input(id("fc_fix"), "value"),
        Input(id("mu_fix"), "value"),
        Input(id("wf_scale"), "value"),
        Input(id("text_size"), "value"),
        Input(id("wf_linewidth"), "value"),
        Input(id("pe_linewidth"), "value"),
        Input(id("state_linewidth"), "value"),
        Input(id("wf_colour"), "value"),
        Input(id("pe_colour"), "value"),
        Input(id("state_colour"), "value"),
        Input(id("toggle_wf"), "value"),
        Input(id("toggle_pe"), "value"),
        Input(id("toggle_states"), "value")
    ]
)
def update_app(lin_wn, ang_wn, fc, mu, max_n, lin_wn_fix, ang_wn_fix, fc_fix,
               mu_fix, wf_scale, text_size, wf_linewidth, curve_linewidth,
               state_linewidth, wf_colour, curve_colour, state_colour,
               toggle_wf, toggle_pe, toggle_states):
    """
    Updates the app, given the current state of the UI
    All inputs correspond (in the same order) to those in the decorator
    """

    light = 2.998E10

    # Set all text entries as uneditable
    lin_wn_disable = True
    ang_wn_disable = True
    fc_disable = True
    mu_disable = True

    # Make "fixed" values editable
    if lin_wn_fix:
        lin_wn_disable = False
    if ang_wn_fix:
        ang_wn_disable = False
    if fc_fix:
        fc_disable = False
    if mu_fix:
        mu_disable = False

    # Modebar config

    modebar_options = {
        "modeBarButtonsToRemove": [
            "toImage",
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

    if sum([ang_wn_fix, lin_wn_fix, fc_fix, mu_fix]) != 2 or ang_wn_fix and lin_wn_fix: # noqa

        out_contents = "data:text/csv;charset=utf-8," + urllib.parse.quote(
                ""
            )
        fig = make_subplots()

        rounded = [
            round(lin_wn, 2), round(ang_wn, 2), round(fc, 2), round(mu, 4)
        ]

        on_off = [
            lin_wn_disable, ang_wn_disable, fc_disable, mu_disable
        ]

        return [fig, modebar_options, out_contents] + rounded + on_off  # noqa

    #  Calculate missing parameters
    if ang_wn_fix and not lin_wn_fix:
        omega = ang_wn * light
        lin_wn = ang_wn/(2*np.pi)
        if fc_fix and not mu_fix:
            mu = vibrations.calculate_mu(omega, fc)
        elif not fc_fix and mu_fix:
            fc = vibrations.calculate_k(omega, mu)
    elif lin_wn_fix and not ang_wn_fix:
        omega = lin_wn * 2*np.pi * light
        ang_wn = lin_wn*2*np.pi
        if fc_fix and not mu_fix:
            mu = vibrations.calculate_mu(omega, fc)
        elif not fc_fix and mu_fix:
            fc = vibrations.calculate_k(omega, mu)
    elif not ang_wn_fix and not lin_wn_fix:
        ang_wn = np.sqrt(fc/(mu*1.6605E-27)) / light
        lin_wn = ang_wn/(2*np.pi)

    if max_n is None:
        max_n = 5

    # Read frequency or proxy frequency and convert to angular frequency in s-1
    omega = ang_wn * light
    omega = lin_wn * 2*np.pi * light

    if fc == 0:
        fc = 480

    if omega == 0:
        omega = 2888

    # Convert wavenumbers to frequency in units of s^-1
    state_e, harmonic_e, displacement, zpd = vibrations.calc_harmonic_energies(
        fc,
        mu,
        max_n=max_n
    )

    # Convert to cm-1
    # 1 cm-1 = 1.986 30 x 10-23 J
    state_e /= 1.98630E-23
    harmonic_e /= 1.98630E-23

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if toggle_pe:
        # Plot harmonic energy curve
        fig.add_trace(
            go.Scatter(
                x=displacement*10E10,
                y=harmonic_e,
                hoverinfo="skip",
                line={
                    "width": curve_linewidth,
                    "color": curve_colour
                }
            ),
            secondary_y=False
        )

    if toggle_states:
        # Plot harmonic state energies
        for it, state in enumerate(state_e):
            fig.add_trace(
                go.Scatter(
                    x=displacement*10E10,
                    y=[state]*displacement.size,
                    line={
                        "width": state_linewidth,
                        "color": state_colour
                    },
                    name="n = {}".format(it),
                    hovertemplate="%{x} "+u"\u212B"+" <br>%{y} cm⁻¹<br>"
                ),
                secondary_y=False
            )

    if wf_scale is None:
        wf_scale = 1

    if wf_colour == "tol":
        wf_colour = utils.tol_cols[1:] + utils.wong_cols + utils.def_cols
    elif wf_colour == "wong":
        wf_colour = utils.wong_cols + utils.def_cols + utils.tol_cols[1:]
    elif wf_colour == "normal":
        wf_colour = utils.def_cols + utils.tol_cols[1:] + utils.wong_cols

    wf = np.zeros([max_n+1, displacement.size])

    # Plot harmonic wavefunction
    if toggle_wf:
        for n in range(0, max_n+1):
            wf[n] = vibrations.harmonic_wf(n, displacement*10E10)
            fig.add_trace(
                go.Scatter(
                    x=displacement*10E10,
                    y=wf[n]*(n+1)*wf_scale + n*(lin_wn)+state_e[0],
                    hoverinfo="skip",
                    line={
                        "width": wf_linewidth,
                        "color": wf_colour[n]
                    }
                ),
                secondary_y=False
            )

    fig.update_xaxes(
        autorange=True,
        hoverformat=".3f",
        ticks="outside",
        title={
            "text": "Displacement (" + u"\u212B" + ")",
            "font": {"size": text_size, "color": "black"}
        },
        tickfont={"size": text_size, "color": "black"},
        showticklabels=True,
        showline=True,
        linewidth=1,
        linecolor='black'
    )

    fig.update_yaxes(
        autorange=True,
        hoverformat=".1f",
        title={
            "text": "Energy (cm⁻¹)",
            "font": {"size": text_size, "color": "black"},
            "standoff": 2
        },
        tickfont={"size": text_size, "color": "black"},
        ticks="outside",
        showticklabels=True,
        tickformat="f",
        secondary_y=False,
        showline=True,
        linewidth=1,
        linecolor='black',
    )

    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=60),
        showlegend=False,
        plot_bgcolor="rgb(255, 255,255)"
    )

    # Create output file
    out_contents = create_output_file(displacement, state_e, harmonic_e, wf)

    rounded = [
        round(lin_wn, 2), round(ang_wn, 2), round(fc, 2), round(mu, 4)
    ]

    on_off = [
        lin_wn_disable, ang_wn_disable, fc_disable, mu_disable
    ]

    return [fig, modebar_options, out_contents] + rounded + on_off
