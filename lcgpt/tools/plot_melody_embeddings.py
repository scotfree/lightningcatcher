"""Plot a 3-d run's token embeddings, position embeddings and lm_head rows.

usage: python tools/plot_melody_embeddings.py TAG   (a run from measure_melodies.py)
Each point is shown after rmsnorm (on the sphere of radius sqrt(3)) as the model sees
token+position combined; wte and wpe are also shown raw.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

tag = sys.argv[1]
runs = Path(__file__).resolve().parent.parent / 'data' / 'melody_runs'
w = json.loads((runs / f'{tag}_weights.json').read_text())
names = w['uchars'] + ['BOS']
wte, wpe, lm = np.array(w['wte']), np.array(w['wpe']), np.array(w['lm_head'])

def rms(x):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-5)

fig = plt.figure(figsize=(15, 5))
panels = [('wte (raw)', wte, names), ('wpe (raw)', wpe, [str(i) for i in range(len(wpe))]),
          ('lm_head rows', lm, names)]
for i, (title, pts, labels) in enumerate(panels):
    ax = fig.add_subplot(1, 4, i + 1, projection='3d')
    ax.scatter(*pts.T, c=np.arange(len(pts)), cmap='viridis')
    for p, l in zip(pts, labels):
        ax.text(*p, l, fontsize=8)
    ax.set_title(title)

# rmsnorm(wte[t] + wpe[p]) for p in 0..7: the actual entry points
ax = fig.add_subplot(1, 4, 4, projection='3d')
for t, name in enumerate(names[:-1]):
    pts = rms(wte[t] + wpe[:8])
    ax.plot(*pts.T, '-o', ms=2, lw=0.8)
    ax.text(*pts[0], name, fontsize=8)
ax.set_title('rmsnorm(wte+wpe), pos 0-7')
fig.tight_layout()
fig.savefig(runs / f'{tag}_embeddings.png', dpi=110)

# Cosine similarity of token embeddings, ordered by pitch
order = sorted(range(len(names) - 1), key=lambda i: ('CDEFGAB'.index(names[i].upper()), names[i].islower()))
e = wte[order]
e = e / np.linalg.norm(e, axis=1, keepdims=True)
print('token order:', ''.join(names[i] for i in order))
print(np.array2string(e @ e.T, precision=2, suppress_small=True, max_line_width=200))
