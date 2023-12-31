import logging

import dash
import dash_bootstrap_components as dbc
import torch
from dash import Dash, html
from dotenv import load_dotenv

from ui.config import get_configs

load_dotenv()
configurations = get_configs()
# load expert_gpts.yaml

# read all .yaml files in CONFIGS_PATH and create a list of dicts with the config, the file name and the file path

# external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
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

if torch.cuda.is_available():
    logging.warning("GPU is available")
    logging.warning(f"GPU device number: {torch.cuda.current_device()}")
    logging.warning(f"Number of GPUs available: {torch.cuda.device_count()}")
    logging.warning(
        f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
else:
    logging.warning("GPU is not available")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
