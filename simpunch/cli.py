"""Command line interface."""
import argparse

import toml
from prefect import serve

from simpunch.flow import generate_flow


def main():
    """Simulate PUNCH data with simpunch."""
    parser = argparse.ArgumentParser(prog="punchpipe")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Run the pipline")
    generate_parser.add_argument("config", type=str, help="Path to config for running")

    automate_parser = subparsers.add_parser("automate", help="Run the pipline")
    automate_parser.add_argument("config", type=str, help="Path to config for running")

    args = parser.parse_args()

    if args.command == "generate":
        generate(args.config)
    elif args.command == "automate":
        automate(args.config)
    else:
        parser.print_help()

def generate(configuration_path):
    """Run a single instance of the pipeline."""
    configuration = load_configuration(configuration_path)
    generate_flow(**configuration)

def automate(configuration_path):
    """Automate the data generation using Prefect."""
    configuration = load_configuration(configuration_path)
    serve(generate_flow.to_deployment(name="simulator-deployment",
                                      description="Create more synthetic data.",
                                      cron=configuration.get("cron", "* * * * *")))


def load_configuration(configuration_path: str) -> dict:
    """Load a configuration file."""
    return toml.load(configuration_path)


if __name__ == "__main__":
    generate("/home/marcus.hughes/build4/punch190_simpunch_config.toml")
