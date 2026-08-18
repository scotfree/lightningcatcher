# LightningcatcherGPT

Unpacking Karpathy's minimal, dependency-free GPT (`karpathy_gpt.py`) into a
series of notebooks. See `mini-gpt-lightning-catcher-context.md` for the design,
`CLAUDE.md` for decisions and constraints.

## Setup

Already done in this checkout: `.venv/` (system Python 3.9.6) with jupyterlab,
jupytext, matplotlib and ipykernel, plus a user-level kernel named `lcgpt`
("LightningcatcherGPT"). To rebuild from scratch:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install jupyterlab jupytext matplotlib ipykernel
.venv/bin/python -m ipykernel install --user --name lcgpt --display-name "LightningcatcherGPT"
```

## Running notebooks

The JupyterLab already running on :8888 is rooted at
`~/projects/games/gettingthere` and cannot navigate up to this directory, so
start one here instead:

```sh
.venv/bin/jupyter lab --port 8889
```

Select the **LightningcatcherGPT** kernel. That kernel is registered
user-level, so it also shows up in any other Jupyter server on this machine.

## jupytext

`jupytext.toml` pairs each `.ipynb` with a `.md` (MyST) mirror alongside it.
The pairing happens automatically on save when the notebook is served from this
venv. To sync by hand:

```sh
.venv/bin/jupytext --sync notebooks/*.ipynb
```

Commit both files. Review diffs on the `.md`.

## Data

`data/names.txt` — 32,032 names, longest is 15 characters (which is what
motivates `block_size = 16`). Committed so notebooks work offline; upstream is
`raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt`.

Note `karpathy_gpt.py` reads `input.txt` from the current directory and will
download it if absent. To run it against the committed copy:

```sh
cp data/names.txt input.txt && .venv/bin/python karpathy_gpt.py
```

A full 1000-step run takes roughly 185 seconds (~3 min) on system Python 3.9.

## karpathy_gpt_cli.py

The same algorithm with a command line, for when you want to run it rather than
read it. `karpathy_gpt.py` itself is never edited — the notebooks quote it by
line number — so the conveniences live in a separate file.

```sh
.venv/bin/python karpathy_gpt_cli.py --num-steps 200 --num-docs 5000
.venv/bin/python karpathy_gpt_cli.py --model model.json
```

| flag | meaning |
|---|---|
| `--num-steps N` | training steps (default 1000). Alias: `--training-runs` |
| `--num-docs N` | use only the first N documents after shuffling (default: all) |
| `--model PATH` | load and sample if PATH exists, otherwise train and save there (default `model.json`) |

Two things to know:

- **The tokenizer is saved with the weights.** `uchars` is derived from whichever
  documents were used, so `--num-docs 3` yields a 14-token vocabulary, and
  rebuilding the tokenizer at load time would misalign the ids. The model file
  carries `uchars` and the config, which is also why loading needs no dataset.
- **A model file is loaded in preference to training.** Running with no arguments
  when `model.json` already exists will sample from it rather than retrain.
  Delete it or pass a different `--model` path to train again.
