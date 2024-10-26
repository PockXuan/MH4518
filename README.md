# MH4518 Simulation Techniques in Finances Project

Welcome to the MH4518 Financial Modelling Project!

## Overview

This project aims to develop a financial model for analyzing investment opportunities. It will incorporate various financial metrics and techniques to evaluate the feasibility and profitability of different investment options.

## Installation

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Create a new virtual environment (Optional)

```bash
python -m venv .venv
```

3. Start the virtual environment

```bash
source .venv/bin/activate
```

4. Install the required dependencies by running

```bash
pip install -r requirements.txt
```

## What was implemented thus far

- Geometric Brownian Motion Class

  - Able to simulate M paths for N steps over time T
  - Plotting of the simulations

- Helper functions:
  - Parameter esimation from given dataframe: return $ \mu, {\sigma^2}, v$
  - Update datasets by running
    ```bash
    python update_datasets.py
    ```
