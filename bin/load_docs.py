"""
This file is used to load the documents to vector db
"""

import click

from expert_gpts.llms.main import LLMConfigBuilder
from shared.config import load_config


@click.command()
@click.option("--config", default="configs/mygpt.yaml", help="config file to use")
def load_docs(config):
    config = load_config(config)
    builder = LLMConfigBuilder(config)
    builder.load_docs()


if __name__ == "__main__":
    load_docs()
