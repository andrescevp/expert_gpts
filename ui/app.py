import dash
import dash_bootstrap_components as dbc
from dash import Dash, html
from dotenv import load_dotenv

from ui.config import get_configs

configurations = get_configs()
# load expert_gpts.yaml
load_dotenv()

# read all .yaml files in CONFIGS_PATH and create a list of dicts with the config, the file name and the file path

# external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
)

# https://dash.plotly.com/urls#variable-paths

app.layout = html.Div(
    [
        dbc.NavbarSimple(
            [
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            "Prompt Engineer", href="/user_tools/prompt_engineer"
                        ),
                        dbc.DropdownMenuItem(
                            "LangChain Tools Engineer",
                            href="/user_tools/expert_prompt_to_tool_prompt",
                        ),
                    ],
                    label="User Tools",
                    color="secondary",
                    in_navbar=True,
                ),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            x.config.chain.chain_key, href=f"/expert_chat/{x.key}"
                        )
                        for x in configurations.values()
                    ],
                    label="Expert Chats",
                    color="primary",
                    in_navbar=True,
                    className="ms-2",
                ),
            ],
            brand=[
                dbc.NavbarBrand("Expert GPTs", href="/"),
                dbc.Badge("an easy AI toolbox", color="primary"),
            ],
        ),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
