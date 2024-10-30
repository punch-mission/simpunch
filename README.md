# simpunch

This software creates simulated PUNCH-like data. It's useful in testing the calibration pipeline.

![example simulated image](example.png)

It accepts a total brightness and polarized brightness model as input.
These have been created using the [FORWARD code](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2016.00008/full) from [GAMERA simulation data ](https://arxiv.org/pdf/2405.13069).
These images are fed backward through the pipeline from level 3 to level 0 products,
adding appropriate effects along the way.

## Instructions

1. Install the package with `pip install .`
2. Start prefect with `prefect server start`
3. Create a configuration file like [example_config.toml](example_config.toml)
4. Run with `simpunch generate my_config.toml`
