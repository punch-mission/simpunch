"""Command line interface."""

import click
import toml
from prefect import serve

from simpunch.flow import generate_flow


#@click.group()
def main():
    """Simulate PUNCH data with simpunch."""

#@main.command()
#@click.argument("configuration_path", type=click.Path(exists=True))
def generate(configuration_path):
    """Run a single instance of the pipeline."""
    configuration = load_configuration(configuration_path)
    generate_flow(**configuration)

#@main.command()
def automate():
    """Automate the data generation using Prefect."""
    serve(generate_flow.to_deployment(name="simulator-deployment",
                                      description="Create more synthetic data."))


def load_configuration(configuration_path: str) -> dict:
    """Load a configuration file."""
    return toml.load(configuration_path)

if __name__ == "__main__":
    generate("/home/marcus.hughes/build4/punch190_simpunch_config.toml")
