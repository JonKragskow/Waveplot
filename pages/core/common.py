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
        'tickformat': '%f'
    },
    yaxis={
        'autorange': True,
        'showgrid': False,
        'zeroline': False,
        'showline': True,
        'title_standoff': 20,
        'ticks': 'outside',
        'tickfont': {'family': 'Arial', 'size': 14, 'color': 'black'},
        'showticklabels': True,
        'minor_ticks': 'outside',
        'tickformat': '%f'
    },
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
        'pan3d',
        'autoScale2d',
        'tableRotation',
        'resetCameraLastSave3d'
    ],
    'displaylogo': False
}


def dash_id(page: str) -> callable:
    def func(_id: str):
        return f'{page}_{_id}'
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
        return self.div.value

    @value.setter
    def value(self, value: str):
        self.div.value = self.prefix(value)

    @property
    def id(self) -> str:
        return self.div.id

    @id.setter
    def id(self, value: str):
        self.div.id = self.prefix(value)

    @property
    def label(self) -> str:
        return self.div.label

    @label.setter
    def label(self, value: str):
        self.div.label = value

    @property
    def children(self) -> str:
        return self.div.children

    @children.setter
    def children(self, value: list):
        self.div.children = value


class Container():
    def __init__(self, prefix, **kwargs):
        self.prefix = dash_id(prefix)

        self.container = dbc.Container(
            children=[],
            **kwargs
        )
        return

    @property
    def value(self) -> str:
        return self.container.value

    @value.setter
    def value(self, value: str):
        self.container.value = self.prefix(value)

    @property
    def id(self) -> str:
        return self.container.id

    @id.setter
    def id(self, value: str):
        self.container.id = self.prefix(value)

    @property
    def label(self) -> str:
        return self.container.label

    @label.setter
    def label(self, value: str):
        self.container.label = value

    @property
    def children(self) -> str:
        return self.container.children

    @children.setter
    def children(self, value: list):
        self.container.children = value


class PlotDiv(Div):
    def __init__(self, prefix: callable, layout: go.Layout, config: dict,
                 plot_loading: bool = False, store_loading: bool = False,
                 store_data: str | list | dict = '', **kwargs):
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

        self.error_alert = dbc.Alert(
            id=self.prefix('plot_error'),
            children='',
            is_open=False,
            dismissable=False,
            color='danger',
            style={'text-align': 'center'}
        )

        self.store = dcc.Store(
            id=self.prefix('store'),
            data=store_data
        )

        if store_loading:
            self.storewrapper = [dcc.Loading(self.store, fullscreen=False)]
        else:
            self.storewrapper = [self.store]

        if plot_loading:
            self.plotwrapper = [dcc.Loading(self.plot)]
        else:
            self.plotwrapper = [self.plot]

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
                            self.error_alert,
                            *self.plotwrapper,
                            *self.storewrapper
                        ]
                    )
                ]
            )
        ]

        self.div.children = contents
        return


def make_layout(left_divs: html.Div, right_divs: html.Div) -> html.Div:

    layout = dbc.Container(
        children=[
            dbc.Row([
                dbc.Col(
                    children=[left_divs],
                    sm=12,
                    lg=6
                ),
                dbc.Col(
                    children=[right_divs],
                    sm=12,
                    lg=6
                )
            ]),
            make_footer()
        ],
        fluid=True,
        style={
            'padding-bottom': '50px'
        }
    )
    return layout


def make_footer():

    footer = html.Footer(
        className='footer_custom',
        children=[
            html.Div(
                [
                    dbc.Button(
                        'Jon Kragskow',
                        href='https://www.kragskow.dev',
                        style={
                            'color': '',
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
