"""
This file is used to load the documents to vector db
"""
import os

import click

from mygpt.llms.main import LLMConfigBuilder
from shared.config import load_config

config = load_config(os.getenv("CONFIG_PATH", "mygpt.yaml"))
builder = LLMConfigBuilder(config)


@click.command()
def load_docs():
    builder.load_docs()


if __name__ == "__main__":
    load_docs()
