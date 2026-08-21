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

**Reversed 2026-08-21.** `karpathy.py` is now organised into four labelled
sections, and the notebooks walk them backwards, outputs first:

1. **Output** (`01_output_and_sampling`) — `generate`, sampling, temperature.
2. **Input** (`02_input_and_training`) — tokenizing, the training loop, Adam.
3. **GPT** (`03_the_gpt`) — embeddings, MLP, residual stream, output head.
4. **Attention** (`04_attention`) — lines 120–136, the interior of `gpt`.
5. **Computation** (`05_computation`) — `Value` and `backward()`.

Follow-ups: fine-tuning, deployment, entropy/Markov.

The reason for the order: each notebook establishes the interface
`(config, token_id, pos_id, keys, values) -> logits` and then a later one gives a
better implementation of it. Notebook 01 drives `generate` with an n-gram counter,
notebook 02 with a learnable bigram table, and only notebook 03 introduces the
transformer. `logit_model=` on `train` and `generate` is what makes this possible;
do not remove it.

The old forward-order notebooks are in `notebooks/archive/pre-reorder/`.

## State as of 2026-08-21

All five notebooks are written against the reversed order, quoted by line number,
every source-derived cell verified byte-for-byte, every non-blank source line of
`karpathy.py` 1–279 covered by exactly one section across the set, every section
exactly two cells. Section counts: 6, 12, 8, 8, 6.

Five demos are not from the source and all run: the n-gram counter (§1.6), the
learnable bigram (§2.12), the tiny model (§3.8), attention weights (§4.8) and
gradient descent (§5.6).

**Verbatim source cells are not executable and never have been.** Dedented
fragments reference locals that do not exist, and `generate`'s `logit_model=gpt`
default needs `gpt` at definition time. Only demo cells run. "Run All" raises
`NameError` at §1.4 — that is expected, not a bug.

Regeneration is **not** scripted and never has been (corrected 2026-08-21 — an
earlier version of this file claimed otherwise and sent a session hunting for a
script that does not exist). The notebooks are authored by Claude directly, in a
conversation that first settles the section ordering; the author then edits the
notebooks heavily by hand, so automating past that point is not worth it.

The one technique worth reusing is placeholder substitution *within* an authoring
pass: write `@@SRC a-b@@` into the `.md`, substitute verbatim lines from
`karpathy.py`, convert with jupytext, then verify fidelity and coverage. That
keeps quoted source byte-exact without retyping it. **If `karpathy.py` changes,
every line reference shifts.**

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
2. ~~**Notebook 04 has no tiny-model counterpart.**~~ **Done 2026-08-21.** §4.8
   re-runs lines 111–131 on the 72-parameter model with the attention weights
   kept, printing the triangle. The recomputation was verified against
   `karpathy.gpt` at 0.00e+00 max logit difference. Design doc §7.5 wants
   *heatmaps*; this is a printed table, so the matplotlib version is still open.
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
7. **No word-level demo exists.** The old placeholder section was dropped in the
   reorder rather than carried over. The mechanism exists and is measured
   (61 → 612 vocabulary on the Beatles corpus) and `doc_to_tokens` is covered in
   §2.1; the demo's framing, corpus and scope are still not designed.
8. **The lyrics blob is still fetchable from GitHub by SHA.** History was rewritten
   and force-pushed 2026-08-18, and the file 404s at HEAD, but commit
   `f2a75cd99d5b93a57bd3a1743a38907ac8202492` still resolves via the API. GitHub
   Support must purge it; not yet filed.
9. **Untracked and undecided**: `names_mini3.json` (a saved model; `.gitignore`
   covers `model*.json` but not this name), `notebooks/Untitled.{ipynb,md}`, and
   the six loose `.ipynb` in `notebooks/archive/` (the pre-regeneration set).
   `notebooks/archive/pre-reorder/` *is* tracked — it holds the forward-order
   notebooks, and was kept in a subdirectory because the loose files share its
   filenames and a plain move would have overwritten them.
10. **Dead code in `karpathy.py`**, left deliberately: the leftover section-label
   block at lines 281–293, and `#settings = dict(TRAIN_DEFAULTS)` at line 187
   referring to a name that does not exist.
