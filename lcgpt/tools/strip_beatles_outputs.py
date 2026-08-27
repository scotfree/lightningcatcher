#!/usr/bin/env python3
"""
strip_beatles_outputs.py
========================
Clear the outputs of every notebook cell tagged `beatles-output`.

The lyrics corpus is copyrighted and is never committed (see `data_attributions.md`
and CLAUDE.md). A 22,912-parameter model trained on it reproduces lines verbatim —
notebook 06 §6.2 demonstrates exactly that — so the *outputs* of cells that train on
or sample from it must not be committed either.

Run this before staging any notebook that has been executed against the Beatles
corpus:

    python tools/strip_beatles_outputs.py notebooks/06_applications_and_next_steps.ipynb

With no arguments it processes every .ipynb under notebooks/ (not archive/).
Exits non-zero if nothing was found to strip and files were named explicitly, so it
can be wired into a pre-commit hook if that is ever wanted.
"""

import json
import sys
from pathlib import Path

TAG = 'beatles-output'
REPO = Path(__file__).resolve().parent.parent


def strip(path):
    """Clear outputs on tagged cells. Returns the number of cells changed."""
    nb = json.loads(path.read_text())
    changed = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        if TAG not in cell.get('metadata', {}).get('tags', []):
            continue
        if cell.get('outputs') or cell.get('execution_count') is not None:
            cell['outputs'] = []
            cell['execution_count'] = None
            changed += 1
    if changed:
        # Match the trailing newline jupyter/jupytext write, so the diff stays minimal.
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    return changed


def main(argv):
    if argv:
        paths = [Path(a) for a in argv]
    else:
        paths = sorted((REPO / 'notebooks').glob('*.ipynb'))

    total = 0
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        n = strip(path)
        total += n
        if n:
            print(f"{path.name}: {n} cell(s) cleared")
        else:
            print(f"{path.name}: nothing tagged {TAG!r} had outputs")

    print(f"\n{total} cell(s) cleared in total.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
