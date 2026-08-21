# LightningcatcherGPT

Notebook + podcast series unpacking Karpathy's ~200-line dependency-free GPT
(`karpathy_gpt.py`). Design doc: `mini-gpt-lightning-catcher-context.md`.

## CRITICAL

**The project is "LightningcatcherGPT."** "lightningcatcher" is a Benjamin
Franklin reference. The format it belongs to is **"Lightning Catcher"** (design
doc §1). Earlier drafts said "Lightning Capture" — voice-mode mis-transcription,
corrected 2026-08-18. Don't reintroduce it.

**The notebooks are anchored to `karpathy.py`**, quoted by line number and
verified byte-for-byte. Changing that file shifts every line reference, so re-run
the regeneration rather than hand-patching the notebooks.

**`karpathy_gpt.py` is Karpathy's original, kept unmodified** as the historical
reference. Nothing quotes it any more, but it is what `karpathy.py` descends from.

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
- **Word-level tokenizing now exists in the CLI** (`--token-type word`), which is
  the mechanism the design doc §6.3 demo needs. Measured on
  `data/beatles_first3.txt`: 612-token vocabulary, 22,912 parameters, and the
  truncation problem inverts — 89% of documents overflow `block_size` as letters,
  1 as words. **Notebook 01 §1.8 is still a PLACEHOLDER**: the demo's framing and
  scope are not designed, only the plumbing exists.
- **The lyrics corpus is never committed.** It is copyrighted; `data_attributions.md`
  is the authority. `data/beatles_first3.txt` is gitignored and regenerated locally
  via `get_lcgpt_corpus.py`. It was committed and pushed public once, on
  2026-08-18, and removed by a full history rewrite the same day.
- **The code is split: `karpathy.py` is the algorithm, `lcgpt.py` is everything
  else** (corpus files, save/load, argparse). The notebooks discuss `karpathy.py`
  only. Corpus plumbing therefore dropped out of notebook 01, which is now
  tokenization alone. (Decided 2026-08-20.)
- **Historical: CLI conveniences lived in `karpathy_gpt_cli.py`, not in the anchor.** Argparse
  and checkpointing would shift every line number in `karpathy_gpt.py`, breaking
  56 line references and 35 verbatim code cells across the notebooks. The CLI
  file duplicates the algorithm verbatim and adds `--num-steps`, `--num-docs`
  `--corpus-file`, `--token-type` and `--model`. If the algorithm is ever changed,
  both files must change.
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
- Notebooks need a server rooted **here**; one rooted elsewhere cannot navigate
  up into this directory. See README. A server Claude launches in the background
  dies with the session — start it in a terminal to keep it.
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

Node 1 is now **tokenization only** — corpus plumbing moved to `lcgpt.py`, which
the notebooks do not discuss. Filenames keep the original node names.

## State as of 2026-08-22

All five notebooks are regenerated against `karpathy.py`, quoted by line number,
every source-derived cell verified byte-for-byte, every non-blank source line
covered by exactly one section, every section exactly two cells. Three demos are
not from the source and do run: Shannon n-grams (§1.8), gradient descent (§2.8),
and the tiny model (§3.11).

Regeneration is scripted, not hand-edited: write `@@SRC a-b@@` placeholders into
the `.md`, substitute verbatim lines from `karpathy.py`, convert with jupytext,
then verify fidelity and coverage. **If `karpathy.py` changes, re-run that rather
than patching notebooks** — every line reference shifts.

Notebook demo cells must `sys.path.insert(0, '..')` before importing `karpathy`
or `lcgpt`: Jupyter's working directory is `notebooks/` and the modules are one
level up. Two demos shipped broken for a day because this was only ever tested
with the path injected from outside the cell.

## Open issues

1. **The tiny-model demo's corpus is a fallback.** Two better ideas were tried and
   measured as failures. A copy rule (3rd char = 1st) does not learn reliably even
   at 8000 steps (13/20 samples in-corpus), so it cannot carry a "watch it learn"
   demo. A two-dialect corpus learns well (19/20) but its 2-D token embeddings do
   **not** separate by dialect, so the "plot the embeddings and see the structure"
   payoff is not available at this size. Anything built on embedding geometry
   needs re-measuring first.
2. **Notebook 04 has no tiny-model counterpart.** Attention weights are activations,
   not parameters, and a 1-head model over 4 positions would make them printable.
   Design doc §7.5 wants attention heatmaps; nothing has been built.
3. **`generate()` reseeds the global RNG** (`karpathy.py`, `seed=42` default). Fine
   when every call passes `seed=` explicitly, which the CLI does. A private
   `random.Random(seed)` would keep sampling deterministic without moving anyone
   else's stream.
4. **`new_model_config` calls `random.seed(seed)` unconditionally**, so `seed=None`
   reseeds from OS entropy and discards a seed the caller set beforehand. Guarding
   with `if seed is not None` would let a notebook seed once at the top.
5. **Stale docs in `lcgpt.py`**: the module docstring still describes
   `karpathy_gpt_cli.py`, and its "As a library" paragraph runs into the next
   sentence where examples were removed. `load_docs_textfile`'s docstring still
   promises to fetch a default corpus, which no longer happens by design.
   `SAVED_KEYS` lists `vocab_size` twice; `math` is imported unused.
6. **Bare `python lcgpt.py`** dies with a raw `FileNotFoundError: model.json`
   rather than an argparse usage message. Deliberate that args are required, but
   `--model` still carries a default, which is what makes the message unhelpful.
7. **Notebook 01's word-level section (§1.9) is still a placeholder.** The
   mechanism exists and is measured (61 → 612 vocabulary on the Beatles corpus);
   the demo's framing, corpus and scope are not designed.
8. **The lyrics blob is still fetchable from GitHub by SHA.** History was rewritten
   and force-pushed 2026-08-18, and the file 404s at HEAD, but commit
   `f2a75cd99d5b93a57bd3a1743a38907ac8202492` still resolves via the API. GitHub
   Support must purge it; not yet filed.
9. **Untracked and undecided**: `names_mini3.json` (a saved model; `.gitignore`
   covers `model*.json` but not this name) and `notebooks/archive/` (the
   pre-regeneration notebooks).
