# LightningcatcherGPT

Notebook + podcast series unpacking Karpathy's ~200-line dependency-free GPT
(`karpathy_gpt.py`). Design doc: `mini-gpt-lightning-catcher-context.md`.

## CRITICAL

**The project is "LightningcatcherGPT."** "lightningcatcher" is a Benjamin
Franklin reference. The format it belongs to is **"Lightning Catcher"** (design
doc §1). Earlier drafts said "Lightning Capture" — voice-mode mis-transcription,
corrected 2026-08-18. Don't reintroduce it.

**Never modify `karpathy_gpt.py`.** It is the fixed reference artifact the whole
project is anchored to. Notebooks explain it; they don't edit it.

**Prose is the author's.** Claude supplies runnable code, real LaTeX for the core
math, and skeletons. Any prose Claude writes is minimal and strictly technical —
describing what a specific cell does, nothing more. No narrative voice.

## Decisions & Constraints

- **Lives as a subdirectory** of the `lightningcatcher` repo, committed to
  `master`. Not its own repo. (Decided 2026-08-18.)
- **Code sharing:** the code *under discussion* in a node is written out in full
  in that notebook, so a reader can open any node cold. Only surrounding
  *scaffolding* is factored into a shared module. Reason: "the whole algorithm
  fits in your head" is the point of the project; hiding the subject of the
  lesson behind an import defeats it.
- **Diagrams:** mermaid for structural/architectural pictures (computation graph,
  data flow through the block) — renders natively in JupyterLab 4.1+, and stays a
  readable text diff. matplotlib for anything data-driven (attention heatmaps,
  loss curves). Caveat: mermaid does *not* render in plain `nbconvert` HTML
  export without extra JS — relevant if notebooks get published as static HTML.
- **jupytext pairing** to `md:myst` alongside each `.ipynb` (`jupytext.toml`).
  Reason: prose and LaTeX stay real markdown, so diffs are legible and cells can
  be pasted into the chat surface verbatim.
- **Word-level tokenizer demo (design doc §6.3): Beatles lyrics, PLACEHOLDER
  ONLY.** Corpus and scope are explicitly not yet designed. Do not build it out.
- **CLI conveniences live in `karpathy_gpt_cli.py`, not in the anchor.** Argparse
  and checkpointing would shift every line number in `karpathy_gpt.py`, breaking
  56 line references and 35 verbatim code cells across the notebooks. The CLI
  file duplicates the algorithm verbatim and adds `--num-steps`, `--num-docs`
  `--corpus-file` and `--model`. If the algorithm is ever changed, both files
  must change.
  (Decided 2026-08-18.)
- **Model files are JSON** (`format: lcgpt-1`), carrying config, `uchars` and the
  weights. The tokenizer ships with the weights because `uchars` depends on which
  documents were used, so it cannot be safely rebuilt at load time.
- **Data is committed** (`data/names.txt`, 32,032 names, longest 15 chars —
  which is what motivates `block_size=16`). Reason: notebooks stay reproducible
  offline; the upstream raw.githubusercontent fetch rate-limits.

## Environment

- `.venv/` here — system Python 3.9.6, no Homebrew. jupyterlab, jupytext,
  matplotlib, numpy, ipykernel.
- Registered kernel: **`lcgpt`** / display name "LightningcatcherGPT". User-level
  kernelspec, so any Jupyter server on this machine can see it.
- A JupyterLab 4.5.9 server already runs on **localhost:8888**, but it is rooted
  at `/Users/scotfree/projects/games/gettingthere` and cannot reach this
  directory. See README for how notebooks actually get opened.
- The full 1000-step training run takes ~63s on CPU (~0.06 s/step plus a ~1.2 s
  floor dominated by the 20 inference samples), so notebooks can train live
  rather than shipping cached weights. Runtime scales with step count only —
  `--num-docs` is not a speed lever. An earlier figure of 185s in this file was
  a single unreproducible measurement, corrected 2026-08-18 against repeated
  runs at 200, 400 and 1000 steps.

## Node structure (design doc §5)

1. Plumbing & tokenization  2. Derivatives, computation graphs & gradient
descent  3. The transformer block  4. Attention  5. Loss & training.
Follow-ups: fine-tuning, deployment, entropy/Markov.
