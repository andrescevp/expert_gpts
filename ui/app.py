import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html
from dotenv import load_dotenv

from shared.config import load_config

# load mygpt.yaml
config = load_config("mygpt.yaml")
load_dotenv()
# print(config)
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv"
)

# external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
)
app.layout = html.Div(
    [
        html.H1("MyGpt"),
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page["relative_path"]
                    )
                )
                for page in dash.page_registry.values()
            ]
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
