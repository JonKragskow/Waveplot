import dash
import flask

# external scripts
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
    suppress_callback_exceptions=False,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=0.75"
        }
    ],
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
