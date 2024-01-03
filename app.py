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
import dash
import flask

# external scripts
external_scripts = [
    "https://code.jquery.com/jquery-3.6.3.min.js",
    "https://kit.fontawesome.com/aec084d2c4.js"
]

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" # noqa
    ],
    external_scripts=external_scripts,
    suppress_callback_exceptions=False,
    use_pages=True
)


# Certificate http01 challenge
@app.server.route("/.well-known/acme-challenge/<path:filename>")
def http01_respond(filename):
    return flask.send_file(
        "../../public/.well-known/acme-challenge/{}".format(filename)
    )

# Change header
# %thing% are defined by dash and are replaced at runtime
app.index_string = r"""
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Waveplot</title>
        <meta name="description" content="Online wavefunction viewer">
        <meta name="keywords" content="Online wavefunction viewer">
        <meta name="author" content="Jon Kragskow">
        {%favicon%}
        {%css%}
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-ZLSP2ZF2HN"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-ZLSP2ZF2HN');
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
""" # noqa

server = app.server
