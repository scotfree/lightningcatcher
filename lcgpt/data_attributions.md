# Data Attributions

Provenance and licensing for the corpora used by the Lightning Capture mini-GPT
project. Both files are produced by `get_lcgpt_corpus.py`, which fetches them from
the sources below. This file documents where the data comes from and what you may
do with it — most importantly, which file is safe to commit to a public repo and
which is not.

The data files themselves (`data/names.txt`, `data/beatles_first3.txt`) are plain
one-item-per-line text with no room for inline attribution comments (a comment
line would be read as a training example), so attribution lives here and in the
header of `get_lcgpt_corpus.py`.

---

## 1. `data/names.txt` — first names (character-level demo)

- **Source:** Andrej Karpathy, *makemore* — `names.txt`, pinned commit `988aa59`.
  - https://github.com/karpathy/makemore
  - Raw file: https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
- **Contents:** ~32,033 first names, one per line, lowercase; 26 unique characters;
  longest name 15 characters. This is the exact corpus Karpathy's mini-GPT trains on.
- **License:** MIT (per the makemore repository).
- **Redistribution:** **Safe to commit** to a public repo *with attribution*. Keeping
  a static copy in the repo is fine and makes runs reproducible without a network
  round-trip. Retain this attribution and the MIT license notice.

---

## 2. `data/beatles_first3.txt` — Beatles lyrics, first three UK albums (word-level demo)

- **Source:** Lyrics fetched via the Hugging Face dataset **`cmotions/Beatles_lyrics`**,
  then filtered to the first three UK studio LPs:
  - *Please Please Me* (1963)
  - *With the Beatles* (1963)
  - *A Hard Day's Night* (1964)
  - Dataset: https://huggingface.co/datasets/cmotions/Beatles_lyrics
- **Why this slice:** restricting to the first three albums keeps the word-level
  vocabulary small (~600–700 distinct word-forms), which is what makes the same
  unmodified mini-GPT code tractable on a CPU. (Covers are included by default to
  match the historical figure for these albums; toggle `INCLUDE_COVERS` in the
  script to keep only originals.)
- **Copyright:** ⚠️ **The underlying lyrics are copyrighted** by the respective
  music publishers and rights holders. They are used here strictly for
  **non-commercial, educational/research purposes** — a toy language-model
  demonstration.
- **Redistribution:** **Do NOT commit `data/beatles_first3.txt` to a public repo.**
  Instead:
  - Commit `get_lcgpt_corpus.py` (which contains no lyrics) and this file.
  - Let each user regenerate the corpus locally by running the script.
  - Add the lyrics file to `.gitignore`:
    ```gitignore
    data/beatles_first3.txt
    ```
  - If you publish notebook *output* (e.g. generated samples), keep it to short,
    model-produced snippets rather than reproducing original lyrics.

---

## Regenerating the corpora

```bash
pip install datasets          # needed only for the Beatles step
python get_lcgpt_corpus.py    # writes data/names.txt and data/beatles_first3.txt
```

Flags: `--force` (rebuild existing files), `--names` / `--beatles` (build just one).

> Note on the Beatles step: the exact column/split names of the Hugging Face
> dataset were not verified offline. The script detects the title/album/lyrics
> columns defensively and prints the schema it sees; if it matches too few songs
> it falls back to the full lyrics set with a warning. On first run, glance at the
> printed schema and match count and adjust the column/title logic if needed.
