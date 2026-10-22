"""Measure: does a small model learn melodies, and do its embeddings show structure?

Trains one config on a corpus from abc_to_phrases.py, then reports held-out loss against
unigram and bigram baselines, and how many samples are verbatim training phrases.
Results and weights go to data/melody_runs/ (gitignored: derived from the corpus).

usage: python tools/measure_melodies.py CORPUS TAG [n_embd n_head steps]
    python tools/measure_melodies.py data/melodies_c.txt oct_3d_20k 3 1 20000
"""
import json, math, random, sys, time
from collections import Counter
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
corpus, tag = Path(sys.argv[1]), sys.argv[2]
n_embd, n_head, steps = (int(a) for a in (sys.argv[3:6] or (3, 1, 3000)))
sys.path.insert(0, str(repo))
import karpathy

out = repo / 'data' / 'melody_runs'
out.mkdir(exist_ok=True)

docs = corpus.read_text().split('\n')
docs = [d for d in docs if d]
random.Random(0).shuffle(docs)          # train() walks docs in order; adjacent phrases share a tune
held, train_docs = docs[:500], docs[500:]

config = karpathy.new_model_config(train_docs, seed=42, n_embd=n_embd, n_head=n_head)
t = time.time()
karpathy.train(config, train_docs, num_steps=steps, verbose=False)
train_s = time.time() - t

def nll(doc):
    """Mean per-token loss of one document, as train() computes it."""
    uchars, BOS = config['uchars'], config['BOS']
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = min(config['block_size'], len(tokens) - 1)
    keys, values = [[] for _ in range(config['n_layer'])], [[] for _ in range(config['n_layer'])]
    total = 0.0
    for pos in range(n):
        probs = karpathy.softmax(karpathy.gpt(config, tokens[pos], pos, keys, values))
        total += -math.log(probs[tokens[pos + 1]].data)
    return total / n

# Baselines on the same held-out docs: unigram and bigram (add-one), same truncation.
B = '^'
def seqs(ds):
    for d in ds:
        s = [B] + list(d) + [B]
        yield s[:config['block_size'] + 1]
uni = Counter(c for s in seqs(train_docs) for c in s[1:])
bi = Counter((a, b) for s in seqs(train_docs) for a, b in zip(s, s[1:]))
ctx = Counter(a for s in seqs(train_docs) for a in s[:-1])
V = config['vocab_size']
def base(ds, kind):
    tot = 0.0
    for s in seqs(ds):
        lp = 0.0
        for a, b in zip(s, s[1:]):
            p = (uni[b] + 1) / (sum(uni.values()) + V) if kind == 'uni' else (bi[a, b] + 1) / (ctx[a] + V)
            lp += -math.log(p)
        tot += lp / (len(s) - 1)
    return tot / len(ds)

model_loss = sum(nll(d) for d in held) / len(held)
samples = karpathy.generate(config, num_samples=40, temperature=0.8, verbose=False, seed=1)
train_set = set(train_docs)
result = dict(
    tag=tag, n_embd=n_embd, n_head=n_head, steps=steps, vocab=V,
    params=sum(len(r) for m in config['state_dict'].values() for r in m),
    train_seconds=round(train_s, 1),
    heldout_loss=round(model_loss, 3), unigram=round(base(held, 'uni'), 3), bigram=round(base(held, 'bi'), 3),
    samples_in_corpus=sum(s in train_set for s in samples), samples=samples[:15],
    distinct_phrases=len(train_set), docs=len(train_docs),
)
print(json.dumps(result, indent=1))
(out / f'{tag}.json').write_text(json.dumps(result, indent=1))

wte = [[p.data for p in row] for row in config['state_dict']['wte']]
wpe = [[p.data for p in row] for row in config['state_dict']['wpe']]
lm = [[p.data for p in row] for row in config['state_dict']['lm_head']]
(out / f'{tag}_weights.json').write_text(json.dumps(dict(uchars=config['uchars'], wte=wte, wpe=wpe, lm_head=lm)))
