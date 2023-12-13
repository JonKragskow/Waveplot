from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

BASIC_LAYOUT = go.Layout(
    xaxis={
        'autorange': True,
        'showgrid': False,
        'zeroline': False,
        'showline': True,
        'ticks': 'outside',
        'tickfont': {'family': 'Arial', 'size': 14, 'color': 'black'},
        'showticklabels': True,
        'minor_ticks': 'outside',
        'tickformat': 'digit'
    },
    yaxis={
        'autorange': True,
        'showgrid': False,
        'zeroline': False,
        'title_standoff': 20,
        'showline': True,
        'ticks': 'outside',
        'tickfont': {'family': 'Arial', 'size': 14, 'color': 'black'},
        'showticklabels': True,
        'minor_ticks': 'outside',
        'tickformat': 'digit'
    },
    yaxis2={'title': ''},
    showlegend=True,
    margin=dict(l=90, r=30, t=30, b=60),
    legend={
        'font': {
            'family': 'Arial',
            'size': 12,
            'color': 'black'
        }
    }
)

BASIC_SCENE = {
    'xaxis': {
        'showgrid': False,
        'zeroline': False,
        'showline': False,
        'showticklabels': False,
        'visible': False
    },
    'yaxis': {
        'showgrid': False,
        'zeroline': False,
        'showline': False,
        'showticklabels': False,
        'visible': False
    },
    'zaxis': {
        'showgrid': False,
        'zeroline': False,
        'showline': False,
        'showticklabels': False,
        'visible': False
    },
    'aspectratio': dict(x=1., y=1, z=1.),
    'dragmode': 'orbit'
}

BASIC_CONFIG = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'plot',
        'height': 500,
        'width': 600,
        'scale': 8
    },
    'modeBarButtonsToRemove': [
        'sendDataToCloud',
        'select2d',
        'lasso',
        'zoom3d',
        'pan3d',
        'autoScale2d',
        'tableRotation',
        'orbitRotation',
        'resetCameraLastSave3d'
    ],
    'displaylogo': False
}


def dash_id(page: str):
    def func(_id: str):
        return f"{page}_{_id}"
    return func


class Div():
    def __init__(self, prefix, **kwargs):
        self.prefix = dash_id(prefix)

        self.div = html.Div(
            children=[],
            **kwargs
        )
        return

    @property
    def value(self) -> str:
        return self.tab.value

    @value.setter
    def value(self, value: str):
        self.tab.value = self.prefix(value)

    @property
    def id(self) -> str:
        return self.tab.id

    @id.setter
    def id(self, value: str):
        self.tab.id = self.prefix(value)

    @property
    def label(self) -> str:
        return self.tab.label

    @label.setter
    def label(self, value: str):
        self.tab.label = value

    @property
    def children(self) -> str:
        return self.tab.children

    @children.setter
    def children(self, value: list):
        self.tab.children = value


class PlotDiv(Div):
    def __init__(self, prefix, layout, config, **kwargs):
        # Initialise base class attributes
        super().__init__(prefix=prefix, **kwargs)

        self.plot = dcc.Graph(
            id=self.prefix('2d_plot'),
            className='plot_area',
            mathjax=True,
            figure={
                'data': [],
                'layout': layout
            },
            config=config
        )

        self.store = dcc.Store(
            id=self.prefix('store'),
            data={}
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
                        [
                            self.plot,
                            self.store
                        ]
                    )
                ]
            )
        ]

        self.div.children = contents
        return


def make_layout(left_divs: html.Div, right_divs: html.Div) -> html.Div:

    layout = html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            dbc.Row([
                                dbc.Col(
                                    children=[left_divs],
                                    className="col-6"
                                ),
                                dbc.Col(
                                    children=[right_divs], # noqa
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
            make_footer()
        ]
    )

    return layout


def make_footer():

    footer = html.Footer(
        className="footer_custom",
        children=[
            html.Div(
                [
                    dbc.Button(
                        'Jon Kragskow',
                        href='https://www.kragskow.dev',
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
