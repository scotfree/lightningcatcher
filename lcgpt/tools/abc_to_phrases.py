#!/usr/bin/env python3
"""
abc_to_phrases.py
=================
Build the melody corpus: folk tunes from the Nottingham Music Database, transposed
to the white keys and cut into two-bar phrases, one phrase per line.

Each note is one character, so the anchor's letter tokenizer reads the file as is:

    --tokens octave  (default)  C4..B4 -> CDEFGAB, C5..B5 -> cdefgab   (14 tokens)
    --tokens pitch               every octave -> CDEFGAB                (7 tokens)

Rhythm is ignored: a note is a token whatever its length, and rests are skipped.

Transposition is by key signature, not by tonic, so every mode lands on the white
keys: two sharps (D major, B minor, E dorian, ...) moves down two semitones, and so
on, choosing the shift in [-6, +5] that disturbs the range least. A phrase is kept
only if, after that, every note is a white key -- one accidental drops the phrase,
not the tune. In octave mode a phrase that does not fit C4..B5 is moved by whole
octaves if that makes it fit, and dropped otherwise.

The dataset is GPL-3.0 (jukedeck's cleaned copy of the NMD). It is fetched into
data/nottingham/, which is gitignored, as is the output. Neither is committed.

    python tools/abc_to_phrases.py --fetch
    python tools/abc_to_phrases.py --tokens pitch --out data/melodies_pc.txt
"""

import argparse
import io
import sys
import urllib.request
import warnings
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTTINGHAM = REPO / 'data' / 'nottingham'
ABC_DIR = NOTTINGHAM / 'ABC_cleaned'
COMMIT = '0992bb6cd864f6d4b90d6663e08c0ef53ffaa08f'
ZIP_URL = f'https://github.com/jukedeck/nottingham-dataset/archive/{COMMIT}.zip'

WHITE = {0: 'C', 2: 'D', 4: 'E', 5: 'F', 7: 'G', 9: 'A', 11: 'B'}  # pitch class -> letter
LOW, HIGH = 60, 83                                                  # C4 .. B5, as MIDI
BARS_PER_PHRASE = 2
MIN_NOTES = 4


def fetch():
    """Download the pinned dataset and keep only its cleaned ABC files."""
    if ABC_DIR.exists():
        print(f'{ABC_DIR} already present')
        return
    print(f'fetching {ZIP_URL}')
    data = urllib.request.urlopen(ZIP_URL).read()
    ABC_DIR.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if '/ABC_cleaned/' in name and name.endswith('.abc'):
                (ABC_DIR / Path(name).name).write_bytes(z.read(name))


def shift_for(sharps):
    """Semitones that move a key signature of `sharps` (negative = flats) to none."""
    tonic = (7 * sharps) % 12          # pitch class of the major key with that signature
    shift = -tonic % 12
    return shift - 12 if shift > 5 else shift


def to_token(midi, tokens):
    letter = WHITE[midi % 12]
    if tokens == 'pitch':
        return letter
    return letter if midi < 72 else letter.lower()


def phrase_tokens(midis, tokens):
    """Tokens for one transposed phrase, or (None, reason) if it cannot be written."""
    if any(m % 12 not in WHITE for m in midis):
        return None, 'accidental'
    if tokens == 'octave':
        for octaves in (0, -12, 12, -24, 24):
            if all(LOW <= m + octaves <= HIGH for m in midis):
                midis = [m + octaves for m in midis]
                break
        else:
            return None, 'range'
    return ''.join(to_token(m, tokens) for m in midis), None


def tune_phrases(score, tokens, stats):
    """Yield the phrases of one tune, counting each dropped phrase by reason."""
    measures = [m for m in score.parts[0].getElementsByClass('Measure') if m.number >= 1]
    for i in range(0, len(measures) - BARS_PER_PHRASE + 1, BARS_PER_PHRASE):
        group = measures[i:i + BARS_PER_PHRASE]
        signatures = {m.getContextByClass('KeySignature').sharps for m in group}
        if len(signatures) != 1:
            stats['key change'] += 1
            continue
        shift = shift_for(signatures.pop())
        midis = [n.pitch.midi + shift for m in group for n in m.notes if n.isNote]
        if len(midis) < MIN_NOTES:
            stats['too short'] += 1
            continue
        stats['range low'] = min(stats.get('range low', 999), min(midis))
        stats['range high'] = max(stats.get('range high', 0), max(midis))
        text, reason = phrase_tokens(midis, tokens)
        if text is None:
            stats[reason] += 1
            continue
        stats['kept'] += 1
        yield text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fetch', action='store_true', help='download the dataset first')
    parser.add_argument('--tokens', choices=('octave', 'pitch'), default='octave')
    parser.add_argument('--out', type=Path, default=REPO / 'data' / 'melodies_c.txt')
    args = parser.parse_args(argv)

    if args.fetch:
        fetch()
    if not ABC_DIR.exists():
        sys.exit(f'{ABC_DIR} missing -- run with --fetch')

    warnings.filterwarnings('ignore')
    from music21 import converter  # slow import, and only needed here

    stats = Counter()
    phrases = []
    for path in sorted(ABC_DIR.glob('*.abc')):
        opus = converter.parse(path, format='abc')
        for score in opus.scores:
            stats['tunes'] += 1
            try:
                phrases.extend(tune_phrases(score, args.tokens, stats))
            except Exception as e:  # music21 trips on a few tunes; count them, don't stop
                stats['unparseable tunes'] += 1
        print(f'{path.name}: {len(phrases)} phrases so far', file=sys.stderr)

    args.out.write_text('\n'.join(phrases) + '\n')
    lengths = Counter(len(p) for p in phrases)
    print(f'wrote {len(phrases)} phrases ({len(set(phrases))} distinct) to {args.out}')
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}')
    print(f'  phrase lengths: {dict(sorted(lengths.items()))}')
    print(f'  vocabulary: {"".join(sorted(set("".join(phrases))))}')


if __name__ == '__main__':
    main()
