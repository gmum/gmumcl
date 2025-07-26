# GMUM Continual Learning Framework - Example Usage

## Installation

Install the package from the project root:

```bash
pip install gmumcl
```

## Environment Variables

Set up your environment variables in a `.env` file in the project root. Example:

```env
CACHE_DIR=./cache
DATA_DIR=./data
OUTPUT_DIR=./outputs
```

The example script will automatically load environment variables from `.env` using `python-dotenv` if installed.

## Running

After installing the package and setting up your `.env` and config, do an editable install from the project root:

```bash
pip install .
```

Then run the example from the project root (not from inside `tests/`):

```bash
PYTHONPATH=. HYDRA_FULL_ERROR={0/1} python tests/example.py --config-name example
```
