from dash import html
import dash_bootstrap_components as dbc


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
