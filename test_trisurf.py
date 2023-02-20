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
import numpy.linalg as la
import py3nj
import functools
from scipy.spatial import Delaunay

import dash
from dash import html, Input, Output, dcc, callback
import numpy as np
import xyz_py as xyzp
import xyz_py.atomic as atomic
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sievers_r(theta, phi, a_2, a_4, a_6):

    c_0 = 3./4.*np.pi
    c_2 = a_2 / np.sqrt(4. * np.pi / 5)
    c_4 = a_4 / np.sqrt(4. * np.pi / 9)
    c_6 = a_6 / np.sqrt(4. * np.pi / 13)

    0.25 * np.sqrt(5/np.pi) * (3 * np.cos(theta)**2 - 1)

    # Calculate r, x, y, z values for each theta and phi
    r = c_0
    r += c_2 * 0.25 * np.sqrt(5/np.pi) * (3 * np.cos(theta)**2 - 1)
    r += c_4 * 3/16 * np.sqrt(1/np.pi) * (35 * np.cos(theta)**4 - 30 * np.cos(theta)**2 + 3) # noqa
    r += c_6 * 1/32 * np.sqrt(13/np.pi) * (231 * np.cos(theta)**6 - 315 * np.cos(theta)**4 + 105 * np.cos(theta)**2 - 5) # noqa
    r = r**1./3.

    return r




@functools.lru_cache(maxsize=32)
def wigner3(a, b, c, d, e, f):

    a = int(2*a)
    b = int(2*b)
    c = int(2*c)
    d = int(2*d)
    e = int(2*e)
    f = int(2*f)

    return py3nj.wigner3j(a, b, c, d, e, f)


@functools.lru_cache(maxsize=32)
def wigner6(a, b, c, d, e, f):

    a = int(2*a)
    b = int(2*b)
    c = int(2*c)
    d = int(2*d)
    e = int(2*e)
    f = int(2*f)

    return py3nj.wigner6j(a, b, c, d, e, f)


def compute_a_vals(n, J, mJ, L, S):

    k_max = min(6, int(2*J+1))

    a_vals = []

    if n == 7:
        a_vals = [0., 0., 0.]
    elif n < 7:
        for k in range(2, k_max+2, 2):
            a_vals.append(_compute_light_a_val(n, J, mJ, L, S, k))
    else:
        for k in range(2, k_max+2, 2):
            a_vals.append(_compute_heavy_a_val(J, mJ, n, k))

    return a_vals


def _compute_light_a_val(n, J, mJ, L, S, k):

    a_k = np.sqrt(4*np.pi/(2*k+1))
    a_k *= (-1)**(2*J-mJ+L+S)
    a_k *= 7./(np.sqrt(4*np.pi)) * (2*J + 1) * np.sqrt(2*k+1)
    a_k *= wigner3(J, k, J, -mJ, 0, mJ)/wigner3(L, k, L, -L, 0, L)
    a_k *= wigner6(L, J, S, J, L, k)
    a_k *= wigner3(k, 3, 3, 0, 0, 0)
    summa = 0
    for it in range(1, n+1):
        summa += (-1)**it * wigner3(k, 3, 3, 0, (4-it), (it-4))
    a_k *= summa

    return a_k


def _compute_heavy_a_val(J, mJ, n, k):
    a_k = np.sqrt(4*np.pi/(2*k+1))
    a_k *= (-1)**(J-mJ)
    a_k *= 7/(np.sqrt(4*np.pi))
    a_k *= wigner3(J, k, J, -mJ, 0, mJ)/wigner3(J, k, J, -J, 0, J)
    a_k *= np.sqrt(2*k+1)*wigner3(k, 3, 3, 0, 0, 0)
    summa = 0
    for it in range(1, n-6):
        summa += (-1)**it * wigner3(k, 3, 3, 0, (4-it), (it-4))
    a_k *= summa
    return a_k


def tri_normal(poly):

    cent = poly[0]+poly[1]+poly[2]
    cent /= 3

    a = np.cross(poly[0], poly[1])
    b = np.cross(poly[1], poly[2])
    c = np.cross(poly[2], poly[0])

    m = a + b + c

    if la.norm(m) != 0.:
        mout = m / la.norm(m)
    else:
        mout = m

    return cent


def compute_trisurf(a_2, a_4, a_6):

    # Create angular grid, with values of theta
    phi = np.linspace(0, 2*np.pi, 12)
    theta = np.linspace(0, np.pi, 6)
    u, v = np.meshgrid(phi, theta)
    u = u.flatten()
    v = v.flatten()
    r = sievers_r(v, 0., a_2, a_4, a_6)

    # coordinates of sievers urface
    x = r * np.sin(v)*np.cos(u)
    y = r * np.sin(v)*np.sin(u)
    z = r * np.cos(v)
    vertices = np.array([x, y, z])
    n_vertices = len(x)

    # Points on 2d grid
    points2D = np.vstack([u, v]).T
    tri = Delaunay(points2D)
    verts_to_simp = tri.simplices
    print(np.max(verts_to_simp))

    # Calculate norm of each triangle
    normals = np.array([
        tri_normal(vertices.T[simp])
        for simp in verts_to_simp
    ])

    neighbour_simplices = [
        fns(vit, tri)
        for vit in range(n_vertices)
    ]

    # And then normal of each vertex as mean of normals
    # of each triangle to which it belongs

    vert_normals = np.array([
        np.sum(normals[neigh_simps], axis=0)/len(neigh_simps)
        for neigh_simps in neighbour_simplices
    ])

    return vertices, verts_to_simp, normals


def fns(pindex, triang):
    neighbors = np.array([
        sit
        for sit, simplex in enumerate(triang.simplices)
        if pindex in simplex
    ])
    return neighbors


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
        radii = [atomic.cov_radii[lab] for lab in unique_labs]
    elif radius_type == "atomic":
        radii = [atomic.atomic_radii[lab] for lab in unique_labs]

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


if __name__ == "__main__":

    # external js
    external_scripts = [
        "https://code.jquery.com/jquery-3.6.3.min.js",
        "https://3Dmol.org/build/3Dmol-min.js",
    ]
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" # noqa
        ],
        external_scripts=external_scripts,
    )

    server = app.server

    app.layout = html.Div(children=[
        html.Div(
            id="viewer",
            style={"height": "50vh"}
        ),
        html.Div(
            id="options",
            children=[
                dbc.Button("Click Me", id="the_button", color="primary", className="me-1"),
            ]
        ),
        html.Div(
            id="hidden_div",
            # style={"display": "none"},
            children="hi"
        ),
        dcc.Store(
            id="atom_store"
        ),
        dcc.Store(
            id="style_store"
        ),
        dcc.Store(
            id="vert_store"
        ),
        dcc.Store(
            id="tri_store"
        ),
        dcc.Store(
            id="norm_store"
        )
    ])

    # Clientside callback for javascript molecule viewer
    app.clientside_callback(
        """
        function (atoms_spec, mol_style, vert, tri, norm) {

            let element = document.getElementById("viewer");

            while(element.firstChild){
                element.removeChild(element.firstChild);
            }

            let config = { backgroundOpacity: 0.};
            let viewer = $3Dmol.createViewer(element, config, id="viewer_canv");

            var atoms = eval(atoms_spec)

            var m = viewer.addModel();

            viewer.addCustom(
                {vertexArr:vert, faceArr:tri, normalArr:norm, color:'blue'}
            );

            new_group = []

            viewer.zoomTo();
            viewer.render();

            return new_group;
            }
        """, # noqa
        Output("hidden_div", "children"),
        [
            Input("atom_store", "data"),
            Input("style_store", "data"),
            Input("vert_store", "data"),
            Input("tri_store", "data"),
            Input("norm_store", "data")
        ],
    )

    @callback(
        [
            Output("atom_store", "data"),
            Output("style_store", "data"),
            Output("vert_store", "data"),
            Output("tri_store", "data"),
            Output("norm_store", "data")
        ],
        Input("the_button", "n_clicks")
    )
    def update_mol(_):

        labels, coords = xyzp.load_xyz("assets/nature.xyz")

        labels_nn = xyzp.remove_label_indices(labels)
        adjacency = xyzp.get_adjacency(
            labels,
            coords,
            adjust_cutoff={"Dy": 1.6}
        )

        molecule = make_js_molecule(
            coords, labels_nn, adjacency
        )

        style = make_js_molstyle(
            'stick',
            labels_nn,
            "covalent",
            "m"
        )

        j = 7.5
        mj = 7.5

        a_2 = _compute_heavy_a_val(j, mj, 9, 2)
        a_4 = _compute_heavy_a_val(j, mj, 9, 4)
        a_6 = _compute_heavy_a_val(j, mj, 9, 6)

        vert, v2s, norm = compute_trisurf(a_2, a_4, a_6)

        vert = vert.T

        print(np.shape(vert[:, 1]))
        
        import plotly.figure_factory as ff
        fig = ff.create_trisurf(
            x=vert[:, 0],
            y=vert[:, 1],
            z=vert[:, 2],
            simplices=v2s,
            aspectratio=dict(x=1, y=1, z=1)
        )

        # Turn off plotly gubbins
        layout = go.Layout(
            hovermode=False,
            dragmode="orbit",
            scene_aspectmode="cube",
            scene=dict(
                xaxis=dict(
                    gridcolor="rgb(255, 255, 255)",
                    zerolinecolor="rgb(255, 255, 255)",
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    title="",
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    backgroundcolor="rgb(255, 255,255)",
                    range=[-1,1]
                ),
                yaxis=dict(
                    gridcolor="rgb(255, 255, 255)",
                    zerolinecolor="rgb(255, 255, 255)",
                    showgrid=False,
                    zeroline=False,
                    title="",
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    backgroundcolor="rgb(255, 255,255)",
                    range=[-1,1]
                    ),
                zaxis=dict(
                    gridcolor="rgb(255, 255, 255)",
                    zerolinecolor="rgb(255, 255, 255)",
                    showgrid=False,
                    title="",
                    zeroline=False,
                    showline=False,
                    ticks="",
                    showticklabels=False,
                    backgroundcolor="rgb(255, 255,255)",
                    range=[-1,1]
                ),
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=1
                ),
            ),
            margin={
                "l": 20,
                "r": 30,
                "t": 30,
                "b": 20
            }
        )

        fig.update_layout(layout)

        #fig.show()
        fig = make_subplots()
        # Unblocked rays

        fig.add_trace(
            go.Scatter3d(
                x=vert[:, 0],
                y=vert[:, 1],
                z=vert[:, 2],
                mode="markers",
                showlegend=False,
                marker={"color": 'red'},
            )
        )

        for t in v2s:
            print(t)
            fig.add_trace(
                go.Scatter3d(
                    x=vert[t, 0],
                    y=vert[t, 1],
                    z=vert[t, 2],
                    mode="lines",
                    showlegend=False,
                    marker={"color": 'red'},
                )
            )


        fig.add_trace(
            go.Scatter3d(
                x=norm[:,0],
                y=norm[:,1],
                z=norm[:,2],
                mode="markers",
                showlegend=False,
                marker={"color": 'blue'},
            )
        )

        fig.show()

        vert = vert.tolist()
        v2s = v2s.tolist()
        norm = norm.tolist()

        return molecule, style, vert, v2s, norm

    app.run_server(debug=True, port=8051)
