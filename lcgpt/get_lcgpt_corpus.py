#!/usr/bin/env python3
"""
get_lcgpt_corpus.py
===================
Fetch and prepare the training corpora for the Lightning Capture mini-GPT project.

Running this script produces two plain-text files under ./data/:

    data/names.txt           one name per line          (character-level demo)
    data/beatles_first3.txt  one lyric line per line     (word-level demo)

Both are meant to be *static inputs* to the notebooks so that the model runs are
reproducible. This script exists so the corpora can be regenerated from source at
any time rather than reconstructed by hand.

--------------------------------------------------------------------------------
DATA SOURCES & ATTRIBUTION  (see data_attributions.md for the full version)
--------------------------------------------------------------------------------
1. names.txt
   Source : Andrej Karpathy, "makemore"  (pinned commit 988aa59)
            https://github.com/karpathy/makemore  ->  names.txt
   License: MIT (see the makemore repository). A list of ~32k first names.
            Redistributable with attribution; safe to commit to a public repo.

2. beatles_first3.txt
   Source : Beatles lyrics for the first three UK studio LPs
            (Please Please Me, 1963; With the Beatles, 1963; A Hard Day's Night, 1964),
            fetched via the Hugging Face dataset "cmotions/Beatles_lyrics".
            https://huggingface.co/datasets/cmotions/Beatles_lyrics
   Rights : The underlying lyrics are COPYRIGHTED by the respective music
            publishers / rights holders. They are used here only for
            non-commercial, educational/research purposes (a toy language-model
            demo). >>> Do NOT commit data/beatles_first3.txt to a public repo. <<<
            Commit THIS script instead and let each user regenerate the file
            locally. A ready-to-use .gitignore line is printed after the build.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python get_lcgpt_corpus.py            # build whatever is missing
    python get_lcgpt_corpus.py --force    # rebuild everything
    python get_lcgpt_corpus.py --names    # only the names corpus
    python get_lcgpt_corpus.py --beatles  # only the Beatles corpus

The Beatles step needs the `datasets` library:  pip install datasets
"""

import os
import re
import sys
import urllib.request

DATA_DIR = "data"

NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
NAMES_OUT = os.path.join(DATA_DIR, "names.txt")

BEATLES_HF_DATASET = "cmotions/Beatles_lyrics"
BEATLES_OUT = os.path.join(DATA_DIR, "beatles_first3.txt")

# --------------------------------------------------------------------------------
# The first three UK studio albums. Used to filter the full lyrics set down to a
# small (~600-700 word) vocabulary. Titles are matched after normalization
# (lowercased, punctuation stripped), so exact punctuation here does not matter.
# Several tracks are covers; leave INCLUDE_COVERS = True to match the historical
# "~688 words across the first three albums" figure, or set it False to keep only
# Lennon/McCartney/Harrison originals.
# --------------------------------------------------------------------------------
FIRST_THREE_ALBUMS = {
    "please please me", "with the beatles", "a hard days night",
}

ALBUM_TRACKS = {
    # Please Please Me (1963)
    "I Saw Her Standing There": False, "Misery": False, "Anna (Go to Him)": True,
    "Chains": True, "Boys": True, "Ask Me Why": False, "Please Please Me": False,
    "Love Me Do": False, "P.S. I Love You": False, "Baby It's You": True,
    "Do You Want to Know a Secret": False, "A Taste of Honey": True,
    "There's a Place": False, "Twist and Shout": True,
    # With the Beatles (1963)
    "It Won't Be Long": False, "All I've Got to Do": False, "All My Loving": False,
    "Don't Bother Me": False, "Little Child": False, "Till There Was You": True,
    "Please Mister Postman": True, "Roll Over Beethoven": True, "Hold Me Tight": False,
    "You Really Got a Hold on Me": True, "I Wanna Be Your Man": False,
    "Devil in Her Heart": True, "Not a Second Time": False,
    "Money (That's What I Want)": True,
    # A Hard Day's Night (1964)
    "A Hard Day's Night": False, "I Should Have Known Better": False, "If I Fell": False,
    "I'm Happy Just to Dance with You": False, "And I Love Her": False,
    "Tell Me Why": False, "Can't Buy Me Love": False, "Any Time at All": False,
    "I'll Cry Instead": False, "Things We Said Today": False, "When I Get Home": False,
    "You Can't Do That": False, "I'll Be Back": False,
}

INCLUDE_COVERS = True

# Lines that are structural markers rather than lyrics, e.g. "[Chorus]", "[Verse 1]".
SECTION_RE = re.compile(r"^\s*[\[\(]?\s*(intro|verse|chorus|bridge|outro|refrain|"
                        r"pre-?chorus|solo|instrumental|coda)\b.*$", re.IGNORECASE)
BRACKET_TAG_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")


def _norm(text):
    """Lowercase, drop non-alphanumerics (keep spaces), collapse whitespace."""
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ================================================================================
# 1. NAMES  (Karpathy makemore)
# ================================================================================
def build_names(force=False):
    if os.path.exists(NAMES_OUT) and not force:
        print(f"[names]   {NAMES_OUT} already exists (use --force to rebuild). Skipping.")
        return
    _ensure_data_dir()
    print(f"[names]   downloading from {NAMES_URL}")
    raw = urllib.request.urlopen(NAMES_URL, timeout=60).read().decode("utf-8")
    names = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    with open(NAMES_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    chars = sorted(set("".join(names)))
    print(f"[names]   wrote {NAMES_OUT}: {len(names):,} names, "
          f"{len(chars)} unique chars, longest = {max(map(len, names))}")


# ================================================================================
# 2. BEATLES  (first three albums, via Hugging Face)
# ================================================================================
def _selected_titles():
    """Normalized set of album track titles we want to keep."""
    return {
        _norm(title)
        for title, is_cover in ALBUM_TRACKS.items()
        if INCLUDE_COVERS or not is_cover
    }


def _find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _clean_lyric_lines(lyrics):
    out = []
    for line in str(lyrics).splitlines():
        line = line.strip()
        if not line:
            continue
        if BRACKET_TAG_RE.match(line) or SECTION_RE.match(line):
            continue
        out.append(line)
    return out


def build_beatles(force=False):
    if os.path.exists(BEATLES_OUT) and not force:
        print(f"[beatles] {BEATLES_OUT} already exists (use --force to rebuild). Skipping.")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("[beatles] The `datasets` library is required for this step.\n"
              "[beatles]   pip install datasets\n"
              "[beatles] Skipping Beatles corpus.")
        return

    _ensure_data_dir()
    print(f"[beatles] loading Hugging Face dataset '{BEATLES_HF_DATASET}' ...")
    ds = load_dataset(BEATLES_HF_DATASET)

    # The dataset ships as a DatasetDict with (at least) a plain-lyrics split.
    # Prefer an untagged/full split; otherwise take whatever split exists.
    split_name = next((s for s in ds if "full" in s.lower()), None) or list(ds)[0]
    split = ds[split_name]
    cols = split.column_names
    print(f"[beatles] using split '{split_name}' with columns: {cols}")

    title_col = _find_col(cols, ["title", "song", "name", "track"])
    album_col = _find_col(cols, ["album", "record", "lp"])
    lyric_col = _find_col(cols, ["lyrics", "text", "lyric", "body"])

    if lyric_col is None:
        print(f"[beatles] Could not find a lyrics column in {cols}.\n"
              f"[beatles] Inspect the schema and set `lyric_col` by hand. Skipping.")
        return

    wanted_titles = _selected_titles()

    def keep(row):
        # Prefer album filtering when the column exists; fall back to title match.
        if album_col and row.get(album_col):
            if _norm(row[album_col]) in FIRST_THREE_ALBUMS:
                return True
        if title_col and row.get(title_col):
            if _norm(row[title_col]) in wanted_titles:
                return True
        return False

    rows = [r for r in split if keep(r)]
    print(f"[beatles] matched {len(rows)} songs to the first three albums.")

    # Layered fallback so you always get a usable corpus even if the schema
    # differs from what we assumed. Tighten this once you've seen the real data.
    if len(rows) < 10:
        print("[beatles] WARNING: too few matches -- the schema likely differs from "
              "our assumptions.\n[beatles] Falling back to the FULL lyrics set "
              "(all songs). Cap the vocabulary in the notebook instead, or fix the "
              "column/title matching above.")
        rows = list(split)

    lines = []
    for r in rows:
        lines.extend(_clean_lyric_lines(r[lyric_col]))

    if not lines:
        print("[beatles] No lyric lines extracted -- check the lyrics column. Skipping.")
        return

    with open(BEATLES_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    vocab = {w for ln in lines for w in _norm(ln).split()}
    print(f"[beatles] wrote {BEATLES_OUT}: {len(lines):,} lyric lines, "
          f"~{len(vocab):,} distinct lowercased word-forms.")
    print("[beatles] ------------------------------------------------------------")
    print("[beatles] REMINDER: these lyrics are copyrighted. Do NOT commit")
    print(f"[beatles] {BEATLES_OUT} to a public repo. Add this to .gitignore:")
    print(f"[beatles]     {BEATLES_OUT}")
    print("[beatles] Commit this script; let others regenerate the file locally.")
    print("[beatles] ------------------------------------------------------------")


def main(argv):
    do_names = "--names" in argv
    do_beatles = "--beatles" in argv
    if not do_names and not do_beatles:      # default: do both
        do_names = do_beatles = True
    force = "--force" in argv

    if do_names:
        build_names(force=force)
    if do_beatles:
        build_beatles(force=force)


if __name__ == "__main__":
    main(sys.argv[1:])
